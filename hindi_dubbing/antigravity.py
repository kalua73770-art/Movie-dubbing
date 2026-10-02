from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
from pathlib import Path

from google import genai
from google.genai import types


class AntigravityService:
    """Movie-level continuity reasoning agent."""

    def __init__(self, settings, log=None):
        self.s = settings
        self.log = log or logging.getLogger("antigravity")
        self.keys = list(self.s.antigravity_api_keys)
        self.clients = {}
        self.active_key = None

    def client(self, key):
        if key not in self.clients:
            self.clients[key] = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=int(self.s.antigravity_timeout_ms),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        return self.clients[key]

    @staticmethod
    def parse_json(text):
        text = (text or "").strip()
        text = re.sub(r"^\s*\`\`\`(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*\`\`\`\s*$", "", text)
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("Antigravity returned no JSON object")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("Antigravity returned non-object JSON")
        return data

    @staticmethod
    def frame(video, seconds, out):
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            return out
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{max(0.0, seconds):.3f}",
                "-i", str(video), "-frames:v", "1", "-q:v", "3", str(out)
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return out if out.exists() else None

    @staticmethod
    def windows(segments, step, maximum):
        if not segments:
            return []
        duration = max(float(s["end"]) for s in segments)
        raw = []
        start = 0.0
        while start < duration:
            end = min(duration, start + max(30, int(step)))
            raw.append((start, end))
            start = end
        if len(raw) <= maximum:
            return raw
        picks = [
            round(i * (len(raw) - 1) / max(1, maximum - 1))
            for i in range(maximum)
        ]
        return [raw[i] for i in dict.fromkeys(picks)]

    def call(self, prompt, images, previous_id):
        if not self.keys:
            raise RuntimeError("ANTIGRAVITY_API_KEYS is empty")
        order = list(range(len(self.keys)))
        if self.active_key in order:
            order.remove(self.active_key)
            order.insert(0, self.active_key)

        last = None
        for key_index in order[:max(1, int(self.s.antigravity_max_keys))]:
            try:
                client = self.client(self.keys[key_index])
                inputs = [{"type": "text", "text": prompt}]
                for image in images:
                    inputs.append({
                        "type": "image",
                        "data": base64.b64encode(image.read_bytes()).decode("ascii"),
                        "mime_type": "image/jpeg",
                    })

                params = {
                    "agent": self.s.antigravity_agent,
                    "input": inputs,
                    "environment": "remote",
                    "agent_config": {
                        "type": "antigravity",
                        "max_total_tokens": int(self.s.antigravity_max_total_tokens),
                    },
                    "timeout": max(1, int(self.s.antigravity_timeout_ms / 1000)),
                }
                if previous_id:
                    params["previous_interaction_id"] = previous_id

                interaction = client.interactions.create(**params)
                output = getattr(interaction, "output_text", "") or ""
                if not output:
                    raise RuntimeError("Antigravity returned empty output")

                self.active_key = key_index
                return self.parse_json(output), str(getattr(interaction, "id", "") or ""), key_index
            except Exception as exc:
                last = exc
                self.log.warning("Antigravity key #%d failed: %s", key_index + 1, str(exc)[:900])

        raise RuntimeError(f"Antigravity failed on available keys: {last}")

    def analyze(self, video, segments, brain, work):
        windows = self.windows(
            segments,
            int(self.s.antigravity_window_seconds),
            int(self.s.antigravity_max_windows),
        )
        result = {
            "annotations": [],
            "characters": {},
            "scene_summaries": [],
            "relationships": [],
            "glossary": {},
            "important_events": [],
            "decisions": [],
        }
        previous_id = brain.interaction_id("antigravity")

        for index, (start, end) in enumerate(windows):
            current = [
                s for s in segments
                if float(s["end"]) > start and float(s["start"]) < end
            ]
            if not current:
                continue

            images = []
            span = max(0.0, end - start)
            for part, ratio in enumerate((0.3, 0.75)):
                image = self.frame(
                    video,
                    start + span * ratio,
                    work / "antigravity_frames" / f"scene_{index:03d}_{part}.jpg",
                )
                if image:
                    images.append(image)

            transcript = "\n".join(
                f'{s["id"]} | {s["start"]:.2f}-{s["end"]:.2f}s | '
                f'speaker={s.get("speaker")} | {s["text"]}'
                for s in current
            )[:14000]

            context = (
                brain.agent_context(35000)
                if not previous_id
                else brain.context_for_segments(current)
            )
            prompt = (
                "Act as the movie's canonical continuity director. "
                "Use the scene images and timed transcript. MovieBrain is authoritative. "
                "Never invent unsupported facts, never change locked voices, and preserve "
                "existing character_id values. Use unknown or ambiguous when uncertain. "
                "Return JSON with keys characters, annotations, scene_summary, relationships, "
                "glossary, important_events, decisions. "
                "Each annotation must include segment_id, character_id, confidence, emotion, "
                "pace, intensity, style, on_screen. "
                "Each character must include character_id, name, gender, age_group, "
                "voice_profile, personality, confidence.\n\n"
                f"Scene: {start:.2f}-{end:.2f}s\n"
                "Canonical memory:\n" + context +
                "\n\nTranscript:\n" + transcript
            )

            data, interaction_id, key_index = self.call(
                prompt,
                images,
                previous_id,
            )
            if interaction_id:
                previous_id = interaction_id
                brain.set_interaction_id("antigravity", interaction_id)
                brain.set_interaction_id("antigravity_key", str(key_index))

            accepted = brain.apply_agent_update(
                data,
                segment_ids={str(s["id"]) for s in current},
            )
            data["annotations"] = accepted

            chars = data.get("characters") or {}
            if isinstance(chars, list):
                chars = {
                    str(c.get("character_id")): c
                    for c in chars
                    if isinstance(c, dict) and c.get("character_id")
                }
            for cid, profile in chars.items():
                result["characters"].setdefault(cid, {}).update(profile)

            result["annotations"].extend(accepted)
            result["relationships"].extend(data.get("relationships") or [])
            result["glossary"].update(data.get("glossary") or {})
            result["important_events"].extend(data.get("important_events") or [])
            result["decisions"].extend(data.get("decisions") or [])
            if data.get("scene_summary"):
                result["scene_summaries"].append({
                    "start": start,
                    "end": end,
                    "summary": str(data["scene_summary"])[:1200],
                })

            brain.save()
            self.log.info(
                "Antigravity scene %d/%d complete %.2f-%.2fs key #%d",
                index + 1,
                len(windows),
                start,
                end,
                key_index + 1,
            )

        return result
