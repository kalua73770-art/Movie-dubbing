from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Iterable

from google import genai
from google.genai import types


class GeminiService:
    """Resilient Gemini client with task-aware model/key fallback.

    The router deliberately keeps separate pools for text, transcription and TTS.
    A normal text model is never used as a replacement for transcription or TTS.
    """

    def __init__(self, settings, log=None):
        self.s = settings
        self.log = log or logging.getLogger("gemini")
        self.s.validate()
        self._lock = threading.RLock()
        self._clients: dict[tuple[str, int], genai.Client] = {}
        self._cooldowns: dict[tuple[str, str, str], float] = {}
        self._last_good: dict[tuple[str, str], int] = {}
        self._available_keys: dict[tuple[str, str], set[int]] = {}

    def _task_timeout_ms(self, task: str) -> int:
        if task.startswith("tts"):
            return int(self.s.tts_timeout_ms)
        if task == "text":
            return int(self.s.text_timeout_ms)
        if task == "transcribe":
            return int(self.s.transcribe_timeout_ms)
        if task == "video":
            return int(self.s.video_timeout_ms)
        if task == "speaker-analysis":
            return int(self.s.speaker_analysis_timeout_ms)
        return int(self.s.gemini_http_timeout_ms)

    def _client(self, key: str, timeout_ms: int | None = None) -> genai.Client:
        effective_timeout = int(timeout_ms or self.s.gemini_http_timeout_ms)
        cache_key = (key, effective_timeout)
        with self._lock:
            client = self._clients.get(cache_key)
            if client is None:
                client = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(
                        timeout=effective_timeout,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                )
                self._clients[cache_key] = client
            return client

    @staticmethod
    def _plain(obj):
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: GeminiService._plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [GeminiService._plain(v) for v in obj]
        if hasattr(obj, "model_dump"):
            try:
                return GeminiService._plain(obj.model_dump(mode="json"))
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            return GeminiService._plain(vars(obj))
        return str(obj)

    @staticmethod
    def _error_kind(exc: Exception) -> str:
        message = str(exc).lower()
        if any(x in message for x in (
            "429", "resource_exhausted", "quota", "rate limit", "too many requests",
        )):
            return "quota"
        if any(x in message for x in (
            "401", "403", "unauthorized", "permission denied", "api key not valid",
            "invalid api key",
        )):
            return "auth"
        if any(x in message for x in (
            "404", "not found", "requested entity was not found",
            "model not found", "unknown model",
        )):
            return "model_unavailable"
        if any(x in message for x in (
            "400", "invalid_argument", "bad request", "unsupported",
        )):
            return "request"
        if any(x in message for x in (
            "408", "500", "502", "503", "504", "timeout", "timed out",
            "readtimeout", "connecterror", "connection reset", "server disconnected",
            "temporarily unavailable", "unavailable",
        )):
            return "transient"
        return "unknown"

    def _cooldown_seconds(self, kind: str) -> int:
        if kind == "quota":
            return int(self.s.gemini_quota_cooldown_seconds)
        if kind == "auth":
            return int(self.s.gemini_auth_cooldown_seconds)
        if kind == "transient":
            return int(self.s.gemini_transient_cooldown_seconds)
        if kind == "model_unavailable":
            return 86400
        return 0

    def _candidate_keys(
        self,
        task: str,
        model: str,
        preferred_key_index: int | None = None,
    ) -> list[tuple[int, str]]:
        keys = list(self.s.api_keys)
        if not keys:
            return []

        order = list(range(len(keys)))
        with self._lock:
            last = self._last_good.get((task, model))
        preferred = preferred_key_index if preferred_key_index is not None else last
        if preferred is not None and 0 <= preferred < len(keys):
            order.remove(preferred)
            order.insert(0, preferred)

        now = time.monotonic()
        available = self._available_keys.get((task, model)) or self._available_keys.get(("tts", model))
        if available:
            order = [idx for idx in order if idx in available]
        ready = []
        cooling = []
        for idx in order:
            until = self._cooldowns.get((task, model, keys[idx]), 0.0)
            (cooling if until > now else ready).append((idx, keys[idx]))

        # If every key is cooling down, still try them in the same priority order
        # instead of making the entire pipeline wait for the cooldown timer.
        return ready or cooling

    def _record_success(self, task: str, model: str, key_index: int) -> None:
        with self._lock:
            self._last_good[(task, model)] = key_index
            self._cooldowns.pop((task, model, self.s.api_keys[key_index]), None)

    def _record_failure(self, task: str, model: str, key: str, exc: Exception) -> str:
        kind = self._error_kind(exc)
        seconds = self._cooldown_seconds(kind)
        if seconds:
            with self._lock:
                self._cooldowns[(task, model, key)] = time.monotonic() + seconds
        return kind

    def _attempt(
        self,
        *,
        task: str,
        model: str,
        action: Callable[[genai.Client, int, str], object],
        preferred_key_index: int | None = None,
    ):
        last: Exception | None = None
        candidates = self._candidate_keys(task, model, preferred_key_index)
        timeout_ms = self._task_timeout_ms(task)
        for key_index, key in candidates:
            client = self._client(key, timeout_ms=timeout_ms)
            try:
                result = action(client, key_index, key)
                self._record_success(task, model, key_index)
                return result, key_index
            except Exception as exc:
                last = exc
                kind = self._record_failure(task, model, key, exc)
                cooldown = self._cooldown_seconds(kind)
                self.log.warning(
                    "%s model=%s key=#%d failed kind=%s cooldown=%ss: %s",
                    task,
                    model,
                    key_index + 1,
                    kind,
                    cooldown,
                    str(exc)[:900],
                )
                # A model/endpoint 404 is deterministic for this API key/model;
                # do not spend the rest of the key pool repeating it.
                if kind == "model_unavailable":
                    break
        raise RuntimeError(
            f"{task} failed for model={model} on all configured API keys: {last}"
        )

    def probe_models(self, task_models: dict[str, list[str]]) -> dict[str, list[str]]:
        """
        Cheap capability/endpoint preflight using models.get().
        This confirms that the configured model is exposed to at least one API key.
        It does not pretend to predict transient backend capacity; 429/503 is still
        handled by the runtime circuit breaker.
        """
        health = {}
        self._available_keys: dict[tuple[str, str], set[int]] = {}
        for task, models in task_models.items():
            healthy = []
            key_pool = self.s.api_keys if task == "tts" else self.s.api_keys[:2]
            for model in models:
                ok = False
                last_error = None
                supported = set()
                for key_index, key in enumerate(key_pool):
                    try:
                        client = self._client(key)
                        info = client.models.get(model=model)
                        display = getattr(info, "display_name", None) or getattr(info, "name", None) or model
                        self.log.info(
                            "Model preflight OK task=%s model=%s key=#%d (%s)",
                            task, model, key_index + 1, display,
                        )
                        ok = True
                        supported.add(key_index)
                    except Exception as exc:
                        last_error = exc
                self._available_keys[("tts" if task == "tts" else task, model)] = supported
                if ok:
                    healthy.append(model)
                else:
                    self.log.warning(
                        "Model preflight unavailable task=%s model=%s: %s",
                        task, model, str(last_error)[:700],
                    )
            health[task] = healthy
            if not healthy:
                self.log.warning("No preflight-ready models for task=%s", task)
        return health

    def start_context_interaction(
        self,
        context_text: str,
        preferred_key_index: int | None = None,
        preferred_model_index: int | None = None,
    ) -> tuple[str, str]:
        models = list(self.s.text_models)
        if models and preferred_model_index is not None:
            offset = preferred_model_index % len(models)
            models = models[offset:] + models[:offset]
        last = None
        for model in models:
            try:
                def action(client, _, __):
                    interaction = client.interactions.create(
                        model=model,
                        input=context_text,
                        timeout=max(1, int(self.s.text_timeout_ms / 1000)),
                    )
                    interaction_id = getattr(interaction, "id", None)
                    if not interaction_id:
                        raise RuntimeError("context interaction returned no id")
                    return str(interaction_id)

                interaction_id, key_index = self._attempt(
                    task="context",
                    model=model,
                    action=action,
                    preferred_key_index=preferred_key_index,
                )
                self.log.info(
                    "Movie Brain context seed model=%s key=#%d id=%s",
                    model, key_index + 1, interaction_id,
                )
                return interaction_id, model
            except Exception as exc:
                last = exc
                self.log.warning("Context seed model=%s failed: %s", model, str(exc)[:700])
        raise RuntimeError(f"Movie Brain context seed failed: {last}")

    def generate_text(
        self,
        prompt,
        preferred_key_index: int | None = None,
        preferred_model_index: int | None = None,
        previous_interaction_id: str | None = None,
    ):
        last: Exception | None = None
        models = list(self.s.text_models)
        if models and preferred_model_index is not None:
            offset = preferred_model_index % len(models)
            models = models[offset:] + models[:offset]
        for model in models:
            try:
                self.log.info("Text model=%s", model)

                def action(client, _, __):
                    if previous_interaction_id:
                        response = client.interactions.create(
                            model=model,
                            input=prompt,
                            previous_interaction_id=previous_interaction_id,
                            timeout=max(1, int(self.s.text_timeout_ms / 1000)),
                        )
                        text = getattr(response, "output_text", None) or getattr(response, "text", None)
                    else:
                        response = client.models.generate_content(
                            model=model,
                            contents=prompt,
                            config={"http_options": {"timeout": int(self.s.text_timeout_ms)}},
                        )
                        text = getattr(response, "text", None)
                    if not text:
                        raise RuntimeError("empty text response")
                    return text

                text, key_index = self._attempt(
                    task="text",
                    model=model,
                    preferred_key_index=preferred_key_index,
                    action=action,
                )
                self.log.info(
                    "Text success model=%s key=#%d",
                    model,
                    key_index + 1,
                )
                return text, model
            except Exception as exc:
                last = exc
                self.log.warning(
                    "Trying next configured text model after %s: %s",
                    model,
                    str(exc)[:700],
                )
        raise RuntimeError(
            f"text generation failed on all configured models/keys: {last}"
        )

    @staticmethod
    def _offset(value) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"-?[0-9.]+", value)
            return float(match.group(0)) if match else 0.0
        if isinstance(value, dict):
            return float(value.get("seconds", 0) or 0) + float(value.get("nanos", 0) or 0) / 1e9
        return 0.0

    @staticmethod
    def _parse_word_annotations(interaction):
        raw = GeminiService._plain(interaction)
        annotations = []

        def walk(value):
            if isinstance(value, dict):
                if value.get("type") == "word_info" or (
                    ("word" in value or "text" in value)
                    and ("start_offset" in value or "startOffset" in value)
                    and ("end_offset" in value or "endOffset" in value)
                ):
                    annotations.append(value)
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(raw)
        return raw, annotations

    def transcribe(self, audio_path: Path, preferred_key_index: int | None = None):
        model = self.s.transcribe_model

        def action(client, key_index, _):
            self.log.info(
                "Transcribing %s model=%s key=#%d",
                audio_path.name,
                model,
                key_index + 1,
            )
            audio_file = client.files.upload(file=str(audio_path))
            try:
                interaction = client.interactions.create(
                    model=model,
                    input=[{
                        "type": "audio",
                        "uri": audio_file.uri,
                        "mime_type": getattr(audio_file, "mime_type", "audio/wav"),
                    }],
                    generation_config={
                        "transcription_config": {
                            "mode": {
                                "type": "verbatim",
                                "diarization_mode": "speaker",
                                "timestamp_granularities": ["word"],
                            }
                        }
                    },
                    timeout=max(1, int(self.s.transcribe_timeout_ms / 1000)),
                )
            finally:
                try:
                    client.files.delete(name=audio_file.name)
                except Exception:
                    pass

            raw, annotations = self._parse_word_annotations(interaction)
            if not annotations:
                self.log.error(
                    "No word annotations for %s. Raw response type=%s",
                    audio_path.name,
                    type(raw).__name__,
                )
                raise RuntimeError("Gemini Transcribe returned no word-level annotations")

            words = []
            for item in annotations:
                text = item.get("word") or item.get("text") or item.get("content") or ""
                if isinstance(text, dict):
                    text = text.get("text", "")
                if not text:
                    continue

                speaker = (
                    item.get("speaker")
                    or item.get("speaker_label")
                    or item.get("speakerLabel")
                    or "spk_1"
                )
                start = item.get("start_offset", item.get("startOffset", 0))
                end = item.get("end_offset", item.get("endOffset", 0))
                words.append({
                    "speaker": str(speaker),
                    "start": self._offset(start),
                    "end": self._offset(end),
                    "text": str(text).strip(),
                })

            words.sort(key=lambda x: (x["start"], x["end"]))
            segments = []
            current = None
            for word in words:
                if current is None:
                    current = {
                        "speaker": word["speaker"],
                        "start": word["start"],
                        "end": word["end"],
                        "words": [word["text"]],
                    }
                    continue

                gap = word["start"] - current["end"]
                too_long = current["end"] - current["start"] >= 14.0
                if word["speaker"] != current["speaker"] or gap > 1.0 or too_long:
                    current["text"] = " ".join(current.pop("words"))
                    segments.append(current)
                    current = {
                        "speaker": word["speaker"],
                        "start": word["start"],
                        "end": word["end"],
                        "words": [word["text"]],
                    }
                else:
                    current["end"] = word["end"]
                    current["words"].append(word["text"])

            if current:
                current["text"] = " ".join(current.pop("words"))
                segments.append(current)

            for i, segment in enumerate(segments):
                segment["id"] = f"local_{i:06d}"

            self.log.info(
                "Transcription success file=%s segments=%d key=#%d",
                audio_path.name,
                len(segments),
                key_index + 1,
            )
            return segments

        result, _ = self._attempt(
            task="transcription",
            model=model,
            action=action,
            preferred_key_index=preferred_key_index,
        )
        return result

    def translate_batch(
        self,
        segments,
        context_text="",
        preferred_key_index=None,
        preferred_model_index=None,
        previous_interaction_id=None,
    ):
        lines = [f'{s["id"]}|||{s["text"]}' for s in segments]
        prompt = (
            "Translate every line below into natural spoken Hindi for movie dubbing. "
            "Preserve meaning, names, emotion and intent. Keep short reactions, interjections, "
            "and expletives natural to the scene. Do not add or remove meaning. "
            "Return exactly one line per input in the format ID|||Hindi text.\n\n"
            + "\n".join(lines)
            + "\n\n" + (context_text or "")
        )
        output, model = self.generate_text(
            prompt,
            preferred_key_index=preferred_key_index,
            preferred_model_index=preferred_model_index,
            previous_interaction_id=previous_interaction_id,
        )
        result = {}
        for line in output.splitlines():
            if "|||" in line:
                key, value = line.split("|||", 1)
                result[key.strip()] = value.strip()
        missing = [s["id"] for s in segments if s["id"] not in result]
        if missing:
            raise RuntimeError(f"translation missing IDs: {missing[:8]}")
        self.log.info("Translated %d segments with %s", len(segments), model)
        return result

    def classify_speaker(self, audio_path: Path, transcript_context: str) -> dict:
        prompt = (
            "You are helping cast a Hindi dub for a fictional movie/anime character. "
            "Analyze the attached speaker audio plus the dialogue context. "
            "Infer the CHARACTER's likely gender presentation, not the human voice actor's gender. "
            "If evidence is weak, choose ambiguous. Return JSON only with exactly these keys: "
            "character_gender (male|female|neutral|ambiguous) and confidence (0 to 1).\n\n"
            f"Dialogue context:\n{transcript_context[:2500]}"
        )
        models = []
        for model in [self.s.speaker_analysis_model, *self.s.text_models]:
            if model and model not in models:
                models.append(model)

        last = None
        for model in models:
            try:
                def action(client, key_index, _):
                    audio_file = None
                    try:
                        self.log.info(
                            "Speaker analysis model=%s key=#%d sample=%s",
                            model, key_index + 1, audio_path.name,
                        )
                        audio_file = client.files.upload(file=str(audio_path))
                        return client.models.generate_content(
                            model=model,
                            contents=[audio_file, prompt],
                            config={"http_options": {"timeout": int(self.s.speaker_analysis_timeout_ms)}},
                        )
                    finally:
                        if audio_file is not None:
                            try:
                                client.files.delete(name=audio_file.name)
                            except Exception:
                                pass

                response, _ = self._attempt(task="speaker-analysis", model=model, action=action)
                text = getattr(response, "text", "") or ""
                match = re.search(r"\{.*\}", text, re.S)
                data = json.loads(match.group(0) if match else text)
                gender = str(data.get("character_gender", "ambiguous")).lower().strip()
                if gender not in {"male", "female", "neutral", "ambiguous"}:
                    gender = "ambiguous"
                try:
                    confidence = float(data.get("confidence", 0.0))
                except Exception:
                    confidence = 0.0
                confidence = max(0.0, min(1.0, confidence))
                if confidence < 0.60 and gender in {"male", "female"}:
                    gender = "ambiguous"
                return {"character_gender": gender, "confidence": confidence, "model": model}
            except Exception as exc:
                last = exc
                self.log.warning("Speaker analysis model=%s failed: %s", model, str(exc)[:800])

        self.log.warning("Speaker analysis failed on all models/keys: %s", last)
        return {"character_gender": "ambiguous", "confidence": 0.0, "model": None}

    def choose_voices(
        self,
        speaker_profiles: dict[str, dict],
        locked_voices: dict[str, str] | None = None,
    ) -> dict[str, str]:
        # Use a broad pool so distinct main characters do not collapse onto the
        # same prebuilt voice merely because the first few configured voices filled up.
        featured = {
            "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
            "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
            "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
            "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird",
            "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
        }
        catalog = {}
        try:
            # One lightweight discovery call is enough; runtime TTS has its own
            # key/model router.
            key = self.s.api_keys[0]
            client = self._client(key)
            response = client.voices.list(type_=["prebuilt"], page_size=1000)
            for voice_obj in response.voices or []:
                voice_id = getattr(voice_obj, "id", None)
                if voice_id in featured:
                    catalog[str(voice_id)] = str(
                        getattr(voice_obj, "gender", None) or "neutral"
                    ).lower()
        except Exception as exc:
            self.log.warning("Voice catalog discovery failed: %s", str(exc)[:600])

        configured = [v for v in self.s.voices if v in featured]
        if not configured:
            configured = ["Kore", "Puck", "Charon", "Fenrir", "Orus", "Leda", "Zephyr", "Aoede"]

        fallback_gender = {
            "Kore": "female", "Leda": "female", "Aoede": "female", "Zephyr": "female",
            "Callirrhoe": "female", "Autonoe": "female", "Despina": "female",
            "Erinome": "female", "Laomedeia": "female", "Achernar": "female",
            "Puck": "male", "Charon": "male", "Fenrir": "male", "Orus": "male",
            "Iapetus": "male", "Algieba": "male", "Algenib": "male",
            "Rasalgethi": "male", "Alnilam": "male", "Gacrux": "male",
            "Sadaltager": "male", "Sulafat": "male",
        }
        for voice_id in configured:
            catalog.setdefault(voice_id, fallback_gender.get(voice_id, "neutral"))

        pools = {
            gender: [
                v for v in configured
                if catalog.get(v, "neutral") == gender
            ]
            for gender in ("male", "female", "neutral")
        }
        for gender in pools:
            if not pools[gender]:
                pools[gender] = [v for v in configured if v not in pools.get("male" if gender != "male" else "female", [])]
                if not pools[gender]:
                    pools[gender] = configured[:]

        result = {}
        locked_voices = locked_voices or {}
        used = set(locked_voices.values())
        for character_id, voice_id in locked_voices.items():
            if character_id in speaker_profiles and voice_id in configured:
                result[character_id] = voice_id
        # Main/declared characters first: more important characters get unique voices.
        ordered = sorted(
            speaker_profiles.items(),
            key=lambda kv: (
                -float(kv[1].get("priority", 0) or 0),
                kv[0],
            ),
        )
        counters = {"male": 0, "female": 0, "neutral": 0}

        for character_id, profile in ordered:
            if character_id in result:
                continue
            gender = str(profile.get("gender", "ambiguous")).lower()
            gender = gender if gender in {"male", "female", "neutral"} else "neutral"
            pool = pools[gender]
            preferred_voice = profile.get("preferred_voice")
            if preferred_voice in pool and preferred_voice not in used:
                chosen = preferred_voice
            else:
                chosen = None
                for voice_id in pool:
                    if voice_id not in used:
                        chosen = voice_id
                        break
                if chosen is None:
                    chosen = pool[counters[gender] % len(pool)]
                    counters[gender] += 1
            result[character_id] = chosen
            used.add(chosen)

        self.log.info("Character voice lock: %s", result)
        return result

    def translate_for_duration(
        self,
        segments,
        preferred_key_index: int | None = None,
        preferred_model_index: int | None = None,
        context_text: str = "",
        previous_interaction_id: str | None = None,
    ):
        lines = []
        for s in segments:
            duration = max(0.25, float(s["end"]) - float(s["start"]))
            pace = s.get("pace", "normal")
            rate = {
                "very_slow": 1.65,
                "slow": 1.95,
                "normal": 2.25,
                "fast": 2.55,
                "very_fast": 2.85,
            }.get(pace, 2.25)
            target_words = max(1, round(duration * rate))
            min_words = max(1, round(target_words * 0.82))
            max_words = max(min_words + 2, round(target_words * 1.18))
            lines.append(
                f'{s["id"]}|||duration={duration:.2f}s|||words={min_words}-{max_words}|||pace={pace}|||source={s["text"]}'
            )
        prompt = (
            "Rewrite each source dialogue into natural spoken Hindi for a professional movie/anime dub. "
            "Preserve ALL meaning, intent, names, relationships, reactions and important details. "
            "Do not omit half of a sentence just to make it short. "
            "Aim for the requested spoken duration and word-count band. "
            "For longer lines, use natural Hindi phrasing, connectives or a brief natural reaction when "
            "needed, but never add unrelated information. For short lines, stay concise. "
            "Return exactly one line per input using ID|||Hindi text.\n\n"
            + "\n".join(lines)
        )
        output, model = self.generate_text(
            prompt,
            preferred_key_index=preferred_key_index,
            preferred_model_index=preferred_model_index,
            previous_interaction_id=previous_interaction_id,
        )
        result = {}
        for line in output.splitlines():
            if "|||" in line:
                key, value = line.split("|||", 1)
                result[key.strip()] = value.strip()
        missing = [s["id"] for s in segments if s["id"] not in result]
        if missing:
            raise RuntimeError(f"translation missing IDs: {missing[:8]}")
        self.log.info(
            "Translated %d segments with duration-aware word targets using %s",
            len(segments), model,
        )
        return result


    def rewrite_for_observed_duration(
        self,
        segment,
        observed_duration: float,
        target_duration: float | None = None,
    ):
        target = max(
            0.25,
            float(target_duration or (float(segment["end"]) - float(segment["start"]))),
        )
        if observed_duration > target:
            direction = (
                "Shorten the Hindi line naturally while preserving meaning and emotion. "
                "Do not remove essential meaning. Use fewer, shorter spoken words."
            )
        else:
            direction = (
                "Expand the Hindi line naturally while preserving meaning and emotion. "
                "Do not add unrelated filler. Use natural Hindi phrasing, connective words, "
                "or a brief natural reaction so the spoken line has enough duration."
            )
        prompt = (
            f"{direction} "
            f"Target spoken duration is about {target:.2f}s and current TTS duration is {observed_duration:.2f}s. "
            "Return only the revised Hindi sentence.\n\n"
            + segment["hindi"]
        )
        output, _ = self.generate_text(prompt)
        revised = output.strip().splitlines()[0].strip() if output.strip() else ""
        if not revised:
            raise RuntimeError("duration rewrite returned empty text")
        return revised

    @staticmethod
    def _style(segment):
        parts = []
        for key in ("emotion", "pace", "intensity", "style", "voice_profile"):
            if segment.get(key):
                parts.append(f"{key}: {segment[key]}")
        return "; ".join(parts) or "natural conversational dubbing performance"

    @staticmethod
    def _save_audio_data(raw: bytes, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if raw[:4] == b"RIFF":
            out_path.write_bytes(raw)
            return
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(raw)

    def tts_segment(
        self,
        segment,
        voice: str,
        out_path: Path,
        preferred_key_index: int | None = None,
        preferred_model_index: int | None = None,
        context_text: str = "",
        previous_interaction_id: str | None = None,
        locked_model: str | None = None,
    ):
        last: Exception | None = None

        # A character's locked model is the preferred/timbre-stable route,
        # not a hard failure boundary. If every key for that model is exhausted
        # or temporarily unavailable, continue through the remaining configured
        # TTS models so one quota outage cannot silence the character/movie.
        all_models = list(self.s.tts_models)
        if locked_model:
            models = [locked_model] + [m for m in all_models if m != locked_model]
        else:
            models = all_models
        if models and preferred_model_index is not None and not locked_model:
            offset = preferred_model_index % len(models)
            models = models[offset:] + models[:offset]

        for model in models:
            def action(client, key_index, _):
                style = self._style(segment)
                self.log.info(
                    "TTS segment=%s model=%s key=#%d voice=%s",
                    segment["id"], model, key_index + 1, voice,
                )
                if model.startswith("gemini-3.8-"):
                    if context_text:
                        style += "; continuity=" + context_text[:500]
                    interaction_kwargs = {
                        "model": model,
                        "input": [{
                            "type": "user_input",
                            "content": [{
                                "type": "text",
                                "text": segment["hindi"],
                                "annotations": [{
                                    "type": "speech_metadata",
                                    "speaker": str(segment.get("speaker", "speaker")),
                                    "style": style,
                                }],
                            }],
                        }],
                        "response_format": {"type": "audio"},
                        "generation_config": {
                            "speech_config": [{"voice": voice}],
                        },
                        "timeout": max(1, int(self.s.tts_timeout_ms / 1000)),
                    }
                else:
                    prompt = (
                        "Perform this exact Hindi dialogue naturally and clearly. "
                        "Do not omit words. Do not add words. "
                        f"{style}.\n{segment['hindi']}"
                    )
                    interaction_kwargs = {
                        "model": model,
                        "input": prompt,
                        "response_format": {"type": "audio"},
                        "generation_config": {
                            "speech_config": [{"voice": voice}],
                        },
                        "timeout": max(1, int(self.s.tts_timeout_ms / 1000)),
                    }
                interaction = client.interactions.create(**interaction_kwargs)
                data = getattr(getattr(interaction, "output_audio", None), "data", None)
                if not data:
                    raise RuntimeError("empty TTS audio response")
                raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
                self._save_audio_data(raw, out_path)
                return out_path

            try:
                self._attempt(
                    task="tts",
                    model=model,
                    action=action,
                    preferred_key_index=preferred_key_index,
                )
                return model
            except Exception as exc:
                last = exc
                self.log.warning(
                    "TTS model=%s exhausted/unavailable; trying next configured model: %s",
                    model,
                    str(exc)[:800],
                )

        raise RuntimeError(f"TTS failed on all configured models/keys: {last}")

    @staticmethod
    def build_tts_input(segments, voices, context_text=""):
        speaker_ids = list(dict.fromkeys(str(s["speaker"]) for s in segments))
        if len(speaker_ids) > 2:
            raise ValueError("Gemini TTS supports at most two speakers per request")

        content = []

        for index, seg in enumerate(segments):
            text = str(seg["hindi"])
            if len(speaker_ids) == 1 and index:
                text = "<short pause> " + text
            style = GeminiService._style(seg)
            if context_text:
                style += "; continuity: " + context_text[:500]
            content.append({
                "type": "text",
                "text": text,
                "annotations": [{
                    "type": "speech_metadata",
                    "speaker": str(seg["speaker"]),
                    "style": style,
                }],
            })

        if len(speaker_ids) == 1:
            speech_config = {
                "speakers": [{
                    "speaker": speaker_ids[0],
                    "voice": voices[speaker_ids[0]],
                }]
            }
        else:
            speech_config = {
                "mode": "conversational",
                "speakers": [
                    {"speaker": sid, "voice": voices[sid]}
                    for sid in speaker_ids
                ],
            }

        return [{"type": "user_input", "content": content}], speech_config

    def tts_model_index(self, model: str) -> int:
        try:
            return self.s.tts_models.index(model)
        except ValueError:
            return 0

    def tts_lanes(self) -> list[tuple[str, int]]:
        """Return model/key lanes that survived TTS preflight.

        Each lane is one concrete model+API-project pairing. The orchestrator
        assigns batches to lanes instead of letting one batch serially probe
        every key/model.
        """
        lanes = []
        available = getattr(self, "_available_keys", {})
        # Interleave models first, then rotate projects/keys. This means
        # consecutive batches use different TTS models instead of hammering
        # every key of one model before trying the next model.
        model_key_pairs = []
        for key_index in range(len(self.s.api_keys)):
            for model in self.s.tts_models:
                supported = available.get(("tts", model), set())
                if not supported or key_index in supported:
                    model_key_pairs.append((model, key_index))
        return model_key_pairs

    def tts_batch_on_lane(
        self,
        segments,
        voices,
        out_path: Path,
        model: str,
        key_index: int,
        context_text="",
    ):
        if key_index < 0 or key_index >= len(self.s.api_keys):
            raise ValueError(f"invalid TTS key index: {key_index}")

        client = self._client(
            self.s.api_keys[key_index],
            timeout_ms=int(self.s.tts_timeout_ms),
        )
        input_value, speech_config = self.build_tts_input(
            segments,
            voices,
            context_text=context_text,
        )
        started = time.monotonic()
        self.log.info(
            "TTS lane START model=%s key=#%d segments=%d speakers=%d",
            model,
            key_index + 1,
            len(segments),
            len({str(s["speaker"]) for s in segments}),
        )
        response = client.interactions.create(
            model=model,
            input=input_value,
            response_format={"type": "audio"},
            generation_config={"speech_config": speech_config},
            timeout=max(1, int(self.s.tts_timeout_ms / 1000)),
        )
        data = getattr(getattr(response, "output_audio", None), "data", None)
        if not data:
            raise RuntimeError("empty TTS audio response")
        raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
        self._save_audio_data(raw, out_path)
        self.log.info(
            "TTS lane DONE model=%s key=#%d elapsed=%.2fs output=%s",
            model, key_index + 1, time.monotonic() - started, out_path.name,
        )
        return out_path


    def tts_batch(
        self,
        segments,
        voices,
        out_path: Path,
        preferred_key_index: int | None = None,
        preferred_model_index: int | None = None,
        context_text: str = "",
    ):
        """Generate several dialogue turns in one TTS request.

        TTS has an input limit of 8,192 tokens on the current 3.8 models, so the
        caller keeps batches bounded by characters/turns. Up to two speakers are
        supported by the TTS API.
        """
        last = None
        models = list(self.s.tts_models)
        if models and preferred_model_index is not None:
            offset = preferred_model_index % len(models)
            models = models[offset:] + models[:offset]

        for model in models:
            def action(client, key_index, _):
                input_value, speech_config = self.build_tts_input(
                    segments,
                    voices,
                    context_text=context_text,
                )
                self.log.info(
                    "TTS batch model=%s key=#%d segments=%d speakers=%d",
                    model,
                    key_index + 1,
                    len(segments),
                    len({str(s["speaker"]) for s in segments}),
                )
                response = client.interactions.create(
                    model=model,
                    input=input_value,
                    response_format={"type": "audio"},
                    generation_config={"speech_config": speech_config},
                    timeout=max(1, int(self.s.tts_timeout_ms / 1000)),
                )
                data = getattr(getattr(response, "output_audio", None), "data", None)
                if not data:
                    raise RuntimeError("empty batch TTS audio response")
                raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
                self._save_audio_data(raw, out_path)
                return out_path

            try:
                self._attempt(
                    task="tts-batch",
                    model=model,
                    action=action,
                    preferred_key_index=preferred_key_index,
                )
                return model
            except Exception as exc:
                last = exc
                self.log.warning(
                    "Trying next batch TTS model after %s: %s",
                    model,
                    str(exc)[:800],
                )

        raise RuntimeError(f"Batch TTS failed on all configured models/keys: {last}")

    def _tts_request(self, client, model, segments, voices, timeout_seconds: int):
        input_value, speech_config = self.build_tts_input(segments, voices)
        return client.interactions.create(
            model=model,
            input=input_value,
            response_format={"type": "audio"},
            generation_config={"speech_config": speech_config},
            timeout=timeout_seconds,
        )

    def tts(self, segments, voices, out_path: Path):
        last: Exception | None = None
        for model in self.s.tts_models:
            def action(client, key_index, _):
                self.log.info(
                    "TTS model=%s key=#%d speakers=%d",
                    model, key_index + 1, len(set(s["speaker"] for s in segments)),
                )
                response = self._tts_request(
                    client,
                    model,
                    segments,
                    voices,
                    max(1, int(self.s.tts_timeout_ms / 1000)),
                )
                data = getattr(getattr(response, "output_audio", None), "data", None)
                if not data:
                    raw = self._plain(response)
                    raise RuntimeError(f"empty TTS audio response: {str(raw)[:500]}")
                raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
                self._save_audio_data(raw, out_path)
                return out_path

            try:
                self._attempt(task="tts-batch", model=model, action=action)
                return model
            except Exception as exc:
                last = exc
                self.log.warning("Trying next batch TTS model after %s: %s", model, str(exc)[:800])

        raise RuntimeError(f"TTS failed on all configured models/keys: {last}")
