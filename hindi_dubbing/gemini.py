from __future__ import annotations

import base64
import json
import logging
import re
import wave
from pathlib import Path

from google import genai


class GeminiService:
    def __init__(self, settings, log=None):
        self.s = settings
        self.log = log or logging.getLogger("gemini")
        self.s.validate()

    def _clients(self):
        for key in self.s.api_keys:
            yield key, genai.Client(api_key=key)

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

    def generate_text(self, prompt):
        last = None
        for model in self.s.text_models:
            for key_no, (key, client) in enumerate(self._clients(), 1):
                try:
                    self.log.info("Text model=%s key=#%d", model, key_no)
                    response = client.models.generate_content(model=model, contents=prompt)
                    text = getattr(response, "text", None)
                    if text:
                        return text, model
                    raise RuntimeError("empty text response")
                except Exception as exc:
                    last = exc
                    self.log.warning("Text model=%s key=#%d failed: %s", model, key_no, str(exc)[:800])
        raise RuntimeError(f"text generation failed on all configured models/keys: {last}")

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

    def transcribe(self, audio_path: Path):
        last = None
        for key_no, (key, client) in enumerate(self._clients(), 1):
            try:
                self.log.info("Transcribing %s with %s key=#%d", audio_path.name, self.s.transcribe_model, key_no)
                audio_file = client.files.upload(file=str(audio_path))
                try:
                    interaction = client.interactions.create(
                        model=self.s.transcribe_model,
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
                    )
                finally:
                    try:
                        client.files.delete(name=audio_file.name)
                    except Exception:
                        pass

                raw = self._plain(interaction)
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
                if not annotations:
                    self.log.error("No word annotations. Raw response keys=%s", list(raw.keys()) if isinstance(raw, dict) else type(raw))
                    raise RuntimeError("Gemini Transcribe returned no word-level annotations")

                words = []
                for item in annotations:
                    text = item.get("word") or item.get("text") or item.get("content") or ""
                    if isinstance(text, dict):
                        text = text.get("text", "")
                    if not text:
                        continue
                    speaker = item.get("speaker") or item.get("speaker_label") or item.get("speakerLabel") or "spk_1"
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
                self.log.info("Transcription returned %d speech segments", len(segments))
                return segments
            except Exception as exc:
                last = exc
                self.log.warning("Transcription key=#%d failed: %s", key_no, str(exc)[:800])

        raise RuntimeError(f"transcription failed on all configured API keys: {last}")

    def translate_batch(self, segments):
        lines = [f'{s["id"]}|||{s["text"]}' for s in segments]
        prompt = (
            "Translate every line below into natural spoken Hindi for movie dubbing. "
            "Preserve meaning, names, emotion and intent. Do not explain anything. "
            "Return exactly one line per input in the format ID|||Hindi text.\n\n"
            + "\n".join(lines)
        )
        output, model = self.generate_text(prompt)
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
        """
        Infer the fictional character's likely gender presentation for voice selection.

        The prompt explicitly distinguishes the character from the human voice actor,
        which matters for anime where a female actor may voice a male character.
        Ambiguous cases are kept neutral instead of forcing a guess.
        """
        prompt = (
            "You are helping cast a Hindi dub for a fictional movie/anime character. "
            "Analyze the attached speaker audio plus the dialogue context. "
            "Infer the CHARACTER's likely gender presentation, not the human voice "
            "actor's gender. Female voice actors may voice male characters and vice versa. "
            "Use names, pronouns, relationships, role/context and the audio as secondary evidence. "
            "If the evidence is weak, choose ambiguous. Return JSON only with exactly these keys: "
            "character_gender (male|female|neutral|ambiguous) and confidence (0 to 1).\n\n"
            f"Dialogue context:\n{transcript_context[:2500]}"
        )
        models = []
        for model in [self.s.speaker_analysis_model, *self.s.text_models]:
            if model and model not in models:
                models.append(model)

        last = None
        for model in models:
            for key_no, (key, client) in enumerate(self._clients(), 1):
                audio_file = None
                try:
                    self.log.info(
                        "Speaker analysis model=%s key=#%d sample=%s",
                        model, key_no, audio_path.name
                    )
                    audio_file = client.files.upload(file=str(audio_path))
                    response = client.models.generate_content(
                        model=model,
                        contents=[audio_file, prompt],
                    )
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
                    self.log.warning(
                        "Speaker analysis model=%s key=#%d failed: %s",
                        model, key_no, str(exc)[:800]
                    )
                finally:
                    if audio_file is not None:
                        try:
                            client.files.delete(name=audio_file.name)
                        except Exception:
                            pass

        self.log.warning("Speaker analysis failed on all models/keys: %s", last)
        return {"character_gender": "ambiguous", "confidence": 0.0, "model": None}

    def choose_voices(self, speaker_profiles: dict[str, dict]) -> dict[str, str]:
        """
        Choose TTS voices using Google's current Voice catalog gender metadata.

        Configured voices are preferred when their catalog metadata matches the
        requested gender. If no configured voice matches, we fall back to other
        prebuilt voices from the catalog. This avoids the old arbitrary
        speaker-1->Kore, speaker-2->Puck cycling.
        """
        catalog = {}
        for key_no, (key, client) in enumerate(self._clients(), 1):
            try:
                response = client.voices.list(type_=["prebuilt"], page_size=1000)
                for voice in response.voices or []:
                    voice_id = getattr(voice, "id", None) or getattr(voice, "display_name", None)
                    gender = getattr(voice, "gender", None)
                    if voice_id:
                        catalog[str(voice_id)] = str(gender or "neutral").lower()
                if catalog:
                    break
            except Exception as exc:
                self.log.warning("Voice catalog lookup key=#%d failed: %s", key_no, str(exc)[:500])

        if not catalog:
            self.log.warning("Voice catalog unavailable; falling back to configured voices")
            return {
                speaker: self.s.voices[i % len(self.s.voices)]
                for i, speaker in enumerate(sorted(speaker_profiles))
            }

        configured = [v for v in self.s.voices if v in catalog]
        fallback_all = list(catalog.keys())
        pools = {}

        for gender in ("male", "female", "neutral"):
            preferred = [v for v in configured if catalog.get(v) == gender]
            if not preferred:
                preferred = [v for v in fallback_all if catalog.get(v) == gender]
            pools[gender] = preferred or fallback_all

        counters = {"male": 0, "female": 0, "neutral": 0}
        used = set()
        result = {}

        for speaker in sorted(speaker_profiles):
            gender = speaker_profiles[speaker].get("character_gender", "ambiguous")
            pool_gender = gender if gender in {"male", "female", "neutral"} else "neutral"
            pool = pools[pool_gender] or fallback_all

            candidate = None
            for _ in range(len(pool)):
                candidate = pool[counters[pool_gender] % len(pool)]
                counters[pool_gender] += 1
                if candidate not in used or len(pool) == 1:
                    break

            result[speaker] = candidate
            used.add(candidate)

        self.log.info("Selected gender-aware voices: %s", result)
        return result

    def _tts_request(self, client, model, segments, voices):
        speaker_ids = list(dict.fromkeys(s["speaker"] for s in segments))
        if len(speaker_ids) > 2:
            raise RuntimeError("Gemini TTS supports at most two speakers per request")

        if len(speaker_ids) == 1:
            prompt = "\n".join(s["hindi"] for s in segments)
            speech_config = [{"voice": voices[speaker_ids[0]]}]
        else:
            names = {speaker_ids[0]: "Speaker 1", speaker_ids[1]: "Speaker 2"}
            prompt = "\n".join(f'{names[s["speaker"]]}: {s["hindi"]}' for s in segments)
            speech_config = [
                {"speaker": "Speaker 1", "voice": voices[speaker_ids[0]]},
                {"speaker": "Speaker 2", "voice": voices[speaker_ids[1]]},
            ]

        return client.interactions.create(
            model=model,
            input=prompt,
            response_format={"type": "audio"},
            generation_config={"speech_config": speech_config},
        )

    def tts(self, segments, voices, out_path: Path):
        last = None
        for model in self.s.tts_models:
            for key_no, (key, client) in enumerate(self._clients(), 1):
                try:
                    self.log.info("TTS model=%s key=#%d speakers=%d", model, key_no, len(set(s["speaker"] for s in segments)))
                    response = self._tts_request(client, model, segments, voices)
                    data = getattr(getattr(response, "output_audio", None), "data", None)
                    if not data:
                        raw = self._plain(response)
                        raise RuntimeError(f"empty TTS audio response: {str(raw)[:500]}")
                    raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    # Gemini 2.5/3.1 preview TTS may return raw 24 kHz 16-bit PCM.
                    if raw[:4] == b"RIFF":
                        out_path.write_bytes(raw)
                    else:
                        with wave.open(str(out_path), "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(24000)
                            wf.writeframes(raw)
                    return model
                except Exception as exc:
                    last = exc
                    self.log.warning("TTS model=%s key=#%d failed: %s", model, key_no, str(exc)[:800])
        raise RuntimeError(f"TTS failed on all configured models/keys: {last}")
