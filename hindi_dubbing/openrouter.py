from __future__ import annotations

import base64
import json
import logging
import mimetypes
import subprocess
import time
from pathlib import Path

import requests


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, transient: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.transient = transient


class OpenRouterService:
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, settings, log=None):
        self.s = settings
        self.log = log or logging.getLogger("openrouter")
        if not self.s.openrouter_api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is missing")

    @staticmethod
    def _parse_json(text: str) -> dict:
        cleaned = (text or "").strip()
        try:
            value = json.loads(cleaned)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
        decoder = json.JSONDecoder()
        for token in ("{", "["):
            idx = cleaned.find(token)
            if idx < 0:
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[idx:])
                if isinstance(value, dict):
                    return value
            except Exception:
                continue
        raise ValueError("OpenRouter returned no valid JSON object")

    @staticmethod
    def _transient(status: int | None, text: str) -> bool:
        if status in {408, 409, 429, 500, 502, 503, 504}:
            return True
        lower = (text or "").lower()
        return any(
            x in lower
            for x in (
                "timeout", "timed out", "temporarily unavailable",
                "rate limit", "too many requests", "overloaded",
                "server disconnected",
            )
        )

    def _chat(self, model: str, messages: list[dict], max_tokens: int, reasoning_effort: str | None = None) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        headers = {
            "Authorization": f"Bearer {self.s.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": "Movie Dubbing Continuity",
        }
        last = None
        for attempt in range(3):
            try:
                response = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=max(30.0, self.s.openrouter_timeout_ms / 1000.0),
                )
                if response.status_code >= 400:
                    raise OpenRouterError(
                        f"HTTP {response.status_code}: {response.text[:1200]}",
                        status_code=response.status_code,
                        transient=self._transient(response.status_code, response.text),
                    )
                choices = response.json().get("choices") or []
                if not choices:
                    raise OpenRouterError("OpenRouter returned no choices")
                text = (choices[0].get("message") or {}).get("content") or ""
                if isinstance(text, list):
                    text = "".join(
                        item.get("text", "") for item in text if isinstance(item, dict)
                    )
                if not text:
                    raise OpenRouterError("OpenRouter returned empty model output")
                return str(text)
            except (requests.RequestException, OpenRouterError) as exc:
                last = exc
                transient = getattr(exc, "transient", True)
                self.log.warning(
                    "OpenRouter %s attempt=%d transient=%s: %s",
                    model, attempt + 1, transient, str(exc)[:900]
                )
                if not transient or attempt >= 2:
                    break
                time.sleep(2 ** attempt)
        raise OpenRouterError(f"OpenRouter request failed for {model}: {last}")

    def smoke_models(self) -> dict:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {self.s.openrouter_api_key}"},
            timeout=max(30.0, self.s.openrouter_timeout_ms / 1000.0),
        )
        if response.status_code >= 400:
            raise OpenRouterError(
                f"OpenRouter models HTTP {response.status_code}: {response.text[:800]}",
                status_code=response.status_code,
            )
        ids = {
            str(item.get("id"))
            for item in (response.json().get("data") or [])
            if isinstance(item, dict)
        }
        required = {self.s.openrouter_video_model, self.s.openrouter_reasoning_model}
        missing = sorted(required - ids)
        if missing:
            raise OpenRouterError(f"OpenRouter models missing: {missing}")
        return {
            "video_model": self.s.openrouter_video_model,
            "reasoning_model": self.s.openrouter_reasoning_model,
        }

    def create_video_chunks(self, source_video: Path, boundaries: list[float], work_dir: Path):
        work_dir.mkdir(parents=True, exist_ok=True)
        chunks = []
        previous = 0.0
        for index, boundary in enumerate(boundaries, 1):
            end = float(boundary)
            if end <= previous + 0.05:
                continue
            duration = end - previous
            video_path = work_dir / f"chunk_{index:03d}_{previous:.3f}_{end:.3f}.mp4"
            audio_path = work_dir / f"chunk_{index:03d}_{previous:.3f}_{end:.3f}.wav"
            video_cmd = [
                "ffmpeg", "-y", "-ss", f"{previous:.3f}", "-t", f"{duration:.3f}",
                "-i", str(source_video),
                "-vf", f"scale={int(self.s.openrouter_video_width)}:-2",
                "-r", str(int(self.s.openrouter_video_fps)),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "29",
                "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart",
                str(video_path),
            ]
            video_result = subprocess.run(video_cmd, capture_output=True, text=True)
            if video_result.returncode != 0 or not video_path.exists():
                raise OpenRouterError(
                    f"OpenRouter video chunk encode failed {previous:.2f}-{end:.2f}s: "
                    f"{video_result.stderr[-700:]}"
                )
            audio_cmd = [
                "ffmpeg", "-y", "-ss", f"{previous:.3f}", "-t", f"{duration:.3f}",
                "-i", str(source_video),
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                str(audio_path),
            ]
            audio_result = subprocess.run(audio_cmd, capture_output=True, text=True)
            if audio_result.returncode != 0 or not audio_path.exists():
                raise OpenRouterError(
                    f"OpenRouter audio chunk encode failed {previous:.2f}-{end:.2f}s: "
                    f"{audio_result.stderr[-700:]}"
                )
            size_mb = video_path.stat().st_size / (1024 * 1024)
            if size_mb > self.s.openrouter_video_max_mb:
                raise OpenRouterError(
                    f"Video chunk {index} is {size_mb:.1f}MB > configured "
                    f"{self.s.openrouter_video_max_mb}MB"
                )
            chunks.append({
                "index": index,
                "start": previous,
                "end": end,
                "video": video_path,
                "audio": audio_path,
                "size_mb": round(size_mb, 2),
            })
            previous = end
        return chunks

    @staticmethod
    def _video_url(path: Path) -> str:
        mime = mimetypes.guess_type(str(path))[0] or "video/mp4"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    @staticmethod
    def _audio64(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def analyze_chunk(
        self,
        video_path: Path,
        audio_path: Path,
        segments: list[dict],
        memory_context: str,
        chunk_start: float,
        chunk_end: float,
    ) -> dict:
        transcript = "\n".join(
            f'{s["id"]} | {s["start"]:.2f}-{s["end"]:.2f}s | '
            f'speaker={s.get("speaker")} | {s.get("text","")}'
            for s in segments
        )
        prompt = (
            "You are the Eyes + Ears of a professional movie/anime Hindi dubbing system. "
            "Use BOTH video and audio plus the timed transcript. Reuse existing fictional "
            "character IDs from MovieBrain. Detect emotion, action, expression, intensity, "
            "pace, on-screen state and scene context. Return ONLY JSON with keys characters, "
            "annotations, observations, scene_summary, relationships, glossary, important_events. "
            "Each annotation needs segment_id, character_id, confidence, emotion, pace, intensity, "
            "style, on_screen. Each character needs character_id, name, gender, age_group, "
            "voice_profile, personality, confidence. Use unknown/ambiguous instead of inventing facts.\n\n"
            f"Chunk {chunk_start:.2f}-{chunk_end:.2f}s\n"
            f"Canonical MovieBrain:\n{memory_context}\n\nTranscript:\n{transcript}"
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": self._video_url(video_path)}},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": self._audio64(audio_path),
                        "format": "wav",
                    },
                },
            ],
        }]
        return self._parse_json(
            self._chat(self.s.openrouter_video_model, messages, 10000, "low")
        )

    def reason_continuity(
        self,
        nano_result: dict,
        segments: list[dict],
        memory_context: str,
        chunk_start: float,
        chunk_end: float,
    ) -> dict:
        prompt = (
            "You are Nemotron Ultra, the canonical continuity brain. MovieBrain is authoritative. "
            "Nano Omni is only the observer. Resolve character identity across chunks, preserve "
            "character_id and locked voice_id, detect contradictions and apply evidence-backed updates only. "
            "Return ONLY JSON with character_aliases, characters, annotations, observations, scene_summary, "
            "relationships, glossary, important_events, decisions. character_aliases maps observer-local IDs "
            "to canonical IDs. Never output or change voice_locks.\n\n"
            f"Chunk {chunk_start:.2f}-{chunk_end:.2f}s\n"
            f"Canonical MovieBrain:\n{memory_context}\n\n"
            f"Nano Omni observations:\n{json.dumps(nano_result, ensure_ascii=False)}\n\n"
            "Transcript:\n"
            + "\n".join(
                f'{s["id"]} | {s["start"]:.2f}-{s["end"]:.2f}s | {s.get("text","")}'
                for s in segments
            )
        )
        return self._parse_json(
            self._chat(
                self.s.openrouter_reasoning_model,
                [{"role": "user", "content": prompt}],
                9000,
                "medium",
            )
        )
