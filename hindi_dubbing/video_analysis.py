from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from pathlib import Path

from google import genai
from google.genai import types


class VideoAnalyzer:
    """Use Gemini video understanding to map dialogue to visible characters and acting."""

    def __init__(self, settings, log: logging.Logger):
        self.s = settings
        self.log = log

    @staticmethod
    def _plain(obj):
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: VideoAnalyzer._plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [VideoAnalyzer._plain(v) for v in obj]
        if hasattr(obj, "model_dump"):
            try:
                return VideoAnalyzer._plain(obj.model_dump(mode="json"))
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            return VideoAnalyzer._plain(vars(obj))
        return str(obj)

    @staticmethod
    def _extract_json(text: str):
        cleaned = text.strip()
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        obj = re.search(r"\{.*\}", cleaned, re.S)
        arr = re.search(r"\[.*\]", cleaned, re.S)
        candidate = obj.group(0) if obj else (arr.group(0) if arr else None)
        if not candidate:
            raise ValueError("Gemini video response did not contain JSON")
        return json.loads(candidate)

    def _clients(self):
        max_keys = min(max(1, int(self.s.video_max_keys_per_model)), len(self.s.api_keys))
        for key_no, key in enumerate(self.s.api_keys[:max_keys], 1):
            yield (
                key_no,
                key,
                genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(
                        timeout=int(self.s.video_timeout_ms),
                    ),
                ),
            )

    def _upload(self, client, video_path: Path):
        media = client.files.upload(file=str(video_path))
        deadline = time.monotonic() + max(
            60.0,
            float(self.s.video_timeout_ms) / 1000.0,
        )
        while time.monotonic() < deadline:
            state = getattr(media, "state", None)
            state_name = getattr(
                state,
                "name",
                str(state) if state else "",
            )
            if state_name in {"ACTIVE", "SUCCEEDED"}:
                return media
            if state_name in {"FAILED", "ERROR"}:
                raise RuntimeError(
                    f"Gemini video file processing failed: {state_name}"
                )
            time.sleep(1.0)
            media = client.files.get(name=media.name)
        raise TimeoutError("Timed out waiting for Gemini video file to become ACTIVE")

    def _analyze_window(
        self,
        client,
        model,
        media,
        window_start,
        window_end,
        transcript_segments,
        registry,
    ):
        transcript = "\n".join(
            f'{s["id"]} | {s["start"]:.2f}-{s["end"]:.2f}s | '
            f'{s["speaker"]} | {s["text"]}'
            for s in transcript_segments
        )
        registry_text = json.dumps(
            registry,
            ensure_ascii=False,
            indent=2,
        )
        prompt = (
            f"You are the supervising director for a high-quality Hindi anime/movie dub.\n"
            f"Inspect ONLY the video period {window_start:.2f}s to {window_end:.2f}s, "
            "but use nearby context when needed.\n"
            "Use the VIDEO: faces, body position, shot composition, mouth/lip movement "
            "when visible, relationships, voice context and scene context.\n"
            "Identify the FICTIONAL CHARACTER, not the human voice actor. A female actor "
            "can voice a male character and vice versa.\n"
            "Reuse an existing character_id when the same fictional character appears again.\n"
            "Return JSON only with keys characters and annotations.\n"
            "Each annotation must contain segment_id, character_id, confidence, emotion, "
            "pace, intensity, style, on_screen.\n"
            "emotion: neutral|happy|sad|angry|fearful|surprised|sarcastic|excited|tired|"
            "whisper|other.\n"
            "pace: very_slow|slow|normal|fast|very_fast. intensity: soft|normal|strong|shout.\n"
            "Each character must contain character_id, name, gender, age_group, "
            "voice_profile, personality. Do not invent unsupported facts; use unknown/ambiguous when evidence is weak.\n"
            "Also return scene_summary, relationships and glossary when supported.\n"
            f"Existing character registry:\n{registry_text}\n"
            f"Timed transcript to annotate:\n{transcript}"
        )
        interaction = client.interactions.create(
            model=model,
            input=[
                {
                    "type": "video",
                    "uri": media.uri,
                    "mime_type": getattr(media, "mime_type", "video/mp4"),
                    "processing": "agentic",
                },
                {"type": "text", "text": prompt},
            ],
            timeout=max(1, int(self.s.video_timeout_ms / 1000)),
        )
        text = getattr(interaction, "output_text", None)
        if not text:
            text = json.dumps(
                self._plain(interaction),
                ensure_ascii=False,
            )
        data = self._extract_json(text)
        if not isinstance(data, dict):
            raise ValueError("Gemini video analysis returned non-object JSON")
        return {
            "characters": data.get("characters") or [],
            "annotations": data.get("annotations") or [],
            "scene_summary": str(data.get("scene_summary") or "").strip(),
            "relationships": data.get("relationships") or [],
            "glossary": data.get("glossary") or {},
        }

    @staticmethod
    def _select_windows(windows, maximum):
        if len(windows) <= maximum:
            return windows
        if maximum == 1:
            return [windows[0]]
        indexes = [
            round(i * (len(windows) - 1) / (maximum - 1))
            for i in range(maximum)
        ]
        selected = []
        seen = set()
        for index in indexes:
            if index not in seen:
                selected.append(windows[index])
                seen.add(index)
        return selected

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(code in message for code in (
            "408", "429", "500", "502", "503", "504",
            "timeout", "timed out", "readtimeout", "connecterror",
            "server disconnected", "temporarily unavailable",
            "resource_exhausted", "too many requests",
        ))


    def _extract_boundary_clip(self, video_path: Path, start: float, end: float, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{max(0.0, start):.3f}", "-to", f"{max(start, end):.3f}",
             "-i", str(video_path), "-vf", "scale=640:-2", "-r", "8",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
             "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(out)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not out.exists():
            raise RuntimeError(f"Gemini boundary clip creation failed: {result.stderr[-500:]}")
        return out

    def suggest_natural_boundaries(
        self, video_path: Path, transcript_segments: list[dict], duration: float,
        target_seconds: int, max_seconds: int, search_seconds: int, work_dir: Path,
    ) -> list[float]:
        if duration <= target_seconds:
            return [round(duration, 3)]
        boundaries, previous = [], 0.0
        model = self.s.video_models[0] if self.s.video_models else None
        if not model:
            return [round(min(duration, target_seconds), 3)]
        ordered = sorted(transcript_segments, key=lambda x: float(x["start"]))
        for boundary_index in range(1, 1000):
            desired = previous + target_seconds
            if desired >= duration or duration - previous <= max_seconds:
                break
            low = max(previous + 30.0, desired - float(search_seconds))
            high = min(duration - 5.0, desired + float(search_seconds))
            gaps = []
            for i in range(len(ordered) - 1):
                left, right = ordered[i], ordered[i + 1]
                gap = float(right["start"]) - float(left["end"])
                midpoint = (float(left["end"]) + float(right["start"])) / 2.0
                if gap >= 0.25 and low <= midpoint <= high:
                    gaps.append((abs(midpoint - desired), midpoint))
            gaps.sort()
            fallback = gaps[0][1] if gaps else desired
            clip_start, clip_end = max(previous, desired - search_seconds), min(duration, desired + search_seconds)
            clip = work_dir / "boundary_checks" / f"boundary_{boundary_index:03d}.mp4"
            try:
                self._extract_boundary_clip(video_path, clip_start, clip_end, clip)
                _, _, client = next(self._clients())
                media = self._upload(client, clip)
                local = "\n".join(
                    f'{s["start"]:.2f}-{s["end"]:.2f}s | {s.get("speaker")} | {s.get("text","")}'
                    for s in ordered if float(s["end"]) > clip_start and float(s["start"]) < clip_end
                )
                prompt = (
                    "Choose the safest cut point. Prefer natural pause, shot transition or speaker turn. "
                    "Never cut inside a word or mid-sentence unless unavoidable. Return only JSON "
                    '{"cut_offset": number}. cut_offset is seconds from this clip start. '
                    f"Preferred target={desired - clip_start:.2f}s; allowed absolute={low:.2f}-{high:.2f}s. "
                    f"Transcript:\n{local}"
                )
                interaction=client.interactions.create(
                    model=model,
                    input=[{"type":"video","uri":media.uri,"mime_type":"video/mp4"},
                           {"type":"text","text":prompt}],
                    timeout=max(30,int(self.s.video_timeout_ms/1000)),
                )
                data=self._extract_json(getattr(interaction,"output_text","") or "")
                candidate=clip_start+float(data.get("cut_offset",0) or 0)
                cut=candidate if low <= candidate <= high else fallback
                self.log.info("Gemini boundary #%d target=%.2f chosen=%.2f",boundary_index,desired,cut)
                try: client.files.delete(name=media.name)
                except Exception: pass
            except Exception as exc:
                cut=fallback
                self.log.warning("Gemini boundary #%d failed; fallback %.2f: %s",boundary_index,cut,str(exc)[:700])
            if cut <= previous + 30.0:
                cut=min(duration-5.0, desired)
            boundaries.append(round(cut,3))
            previous=cut
        boundaries.append(round(duration,3))
        return boundaries

    def analyze(self, video_path: Path, transcript_segments: list[dict]):
        duration = max(
            (float(s["end"]) for s in transcript_segments),
            default=0.0,
        )
        if duration <= 0:
            return {
                "annotations": [],
                "registry": {},
                "scene_summaries": [],
                "relationships": [],
                "glossary": {},
            }

        window_len = max(30, int(self.s.video_analysis_window_seconds))
        windows = []
        start = 0.0
        while start < duration:
            end = min(duration, start + window_len)
            windows.append((start, end))
            start = end

        windows = self._select_windows(
            windows,
            max(1, int(self.s.video_analysis_max_windows)),
        )
        self.log.info(
            "Video analysis using %d windows (max=%d, window=%ss)",
            len(windows),
            self.s.video_analysis_max_windows,
            window_len,
        )

        last_error = None

        for model in self.s.video_models:
            model_completed = False
            for key_no, _, client in self._clients():
                media = None
                try:
                    self.log.info(
                        "Video analysis model=%s key=#%d windows=%d",
                        model,
                        key_no,
                        len(windows),
                    )
                    media = self._upload(client, video_path)

                    annotations = []
                    registry = {}
                    successful_windows = 0
                    scene_summaries = []
                    relationships = []
                    glossary = {}

                    for window_start, window_end in windows:
                        window_segments = [
                            s for s in transcript_segments
                            if float(s["end"]) > window_start
                            and float(s["start"]) < window_end
                        ]
                        if not window_segments:
                            continue

                        self.log.info(
                            "Video analysis model=%s window %.2f-%.2fs segments=%d",
                            model,
                            window_start,
                            window_end,
                            len(window_segments),
                        )
                        try:
                            data = self._analyze_window(
                                client,
                                model,
                                media,
                                window_start,
                                window_end,
                                window_segments,
                                registry,
                            )
                        except Exception as exc:
                            last_error = exc
                            self.log.warning(
                                "Skipping failed video window %.2f-%.2fs: %s",
                                window_start,
                                window_end,
                                str(exc)[:1000],
                            )
                            # One bad window must not throw away the usable
                            # annotations already collected from earlier windows.
                            continue

                        successful_windows += 1
                        for character in data["characters"]:
                            cid = str(character.get("character_id") or "").strip()
                            if cid:
                                registry.setdefault(cid, {}).update({
                                    k: v
                                    for k, v in character.items()
                                    if v not in (None, "")
                                })
                        annotations.extend(
                            a for a in data["annotations"]
                            if a.get("segment_id")
                        )
                        if data.get("scene_summary"):
                            scene_summaries.append({
                                "start": window_start,
                                "end": window_end,
                                "summary": data["scene_summary"],
                            })
                        relationships.extend(data.get("relationships") or [])
                        for k, value in (data.get("glossary") or {}).items():
                            if k not in glossary and value:
                                glossary[str(k)] = str(value)

                    if successful_windows:
                        self.log.info(
                            "Video analysis success model=%s key=#%d "
                            "windows_ok=%d/%d annotations=%d characters=%d",
                            model,
                            key_no,
                            successful_windows,
                            len(windows),
                            len(annotations),
                            len(registry),
                        )
                        return {
                            "annotations": annotations,
                            "registry": registry,
                            "scene_summaries": scene_summaries,
                            "relationships": relationships,
                            "glossary": glossary,
                        }

                    raise RuntimeError(
                        f"No video analysis windows completed for model={model}"
                    )

                except Exception as exc:
                    last_error = exc
                    message = str(exc)
                    self.log.warning(
                        "Video analysis model=%s key=#%d failed: %s",
                        model,
                        key_no,
                        message[:1200],
                    )
                    if self._is_transient(exc):
                        # Move to the next model quickly; do not burn the remaining
                        # keys on an overloaded model.
                        break
                finally:
                    if media is not None:
                        try:
                            client.files.delete(name=media.name)
                        except Exception:
                            pass

            self.log.info(
                "Trying next video model after model=%s",
                model,
            )

        raise RuntimeError(
            f"Gemini video analysis failed on all models/keys: {last_error}"
        )
