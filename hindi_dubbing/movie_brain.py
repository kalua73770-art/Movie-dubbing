from __future__ import annotations

import json
import time
from pathlib import Path


class MovieBrain:
    """Compact persistent continuity state for the whole movie."""

    VERSION = 1
    MAX_SCENE_SUMMARIES = 12
    MAX_GLOSSARY = 80
    MAX_DECISIONS = 80
    MAX_CHARACTERS = 80
    MAX_CONTEXT_CHARS = 6500

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = self._default()

    @staticmethod
    def _default() -> dict:
        return {
            "version": MovieBrain.VERSION,
            "updated_at": time.time(),
            "characters": {},
            "voice_locks": {},
            "relationships": [],
            "scene_summaries": [],
            "glossary": {},
            "decisions": [],
            "interaction_ids": {},
        }

    def load(self) -> "MovieBrain":
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    base = self._default()
                    for key in base:
                        if key in raw:
                            base[key] = raw[key]
                    self.data = base
            except Exception:
                self.data = self._default()
        self.compact(save=False)
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = time.time()
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def snapshot(self) -> dict:
        return json.loads(json.dumps(self.data, ensure_ascii=False))

    def merge_characters(self, registry: dict[str, dict]) -> None:
        chars = self.data.setdefault("characters", {})
        for cid, incoming in (registry or {}).items():
            cid = str(cid).strip()
            if not cid:
                continue
            current = chars.setdefault(
                cid,
                {
                    "character_id": cid,
                    "name": cid,
                    "gender": "ambiguous",
                    "age_group": "unknown",
                    "voice_profile": "",
                    "personality": "",
                    "confidence": 0.0,
                },
            )
            for key in ("name", "gender", "age_group", "voice_profile", "personality"):
                value = incoming.get(key)
                if value not in (None, "") and not current.get(key):
                    current[key] = value
            try:
                current["confidence"] = max(
                    float(current.get("confidence", 0.0) or 0.0),
                    float(incoming.get("confidence", 0.0) or 0.0),
                )
            except Exception:
                pass

    def lock_voices(self, proposed: dict[str, str]) -> dict[str, str]:
        locks = self.data.setdefault("voice_locks", {})
        effective = {}
        for cid, voice in (proposed or {}).items():
            cid = str(cid)
            existing = locks.get(cid)
            if existing:
                effective[cid] = existing
            else:
                locks[cid] = str(voice)
                effective[cid] = str(voice)
        return effective

    def add_relationships(self, values) -> None:
        items = self.data.setdefault("relationships", [])
        seen = {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in items}
        for value in values or []:
            if not isinstance(value, dict):
                continue
            key = json.dumps(value, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                items.append(value)
                seen.add(key)

    def add_scene_summary(self, start: float, end: float, summary: str) -> None:
        summary = str(summary or "").strip()
        if not summary:
            return
        self.data.setdefault("scene_summaries", []).append({
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "summary": summary[:800],
        })
        self.compact(save=False)

    def add_glossary(self, values: dict[str, str]) -> None:
        glossary = self.data.setdefault("glossary", {})
        for key, value in (values or {}).items():
            key = str(key).strip()
            value = str(value).strip()
            if key and value and key not in glossary:
                glossary[key] = value[:300]
        self.compact(save=False)

    def add_decision(self, value: str) -> None:
        value = str(value or "").strip()
        if value:
            self.data.setdefault("decisions", []).append(value[:600])
            self.compact(save=False)

    def set_interaction_id(self, name: str, interaction_id: str | None) -> None:
        if interaction_id:
            self.data.setdefault("interaction_ids", {})[name] = str(interaction_id)

    def interaction_id(self, name: str) -> str | None:
        value = self.data.get("interaction_ids", {}).get(name)
        return str(value) if value else None

    def compact(self, save: bool = True) -> None:
        for key, limit in (
            ("scene_summaries", self.MAX_SCENE_SUMMARIES),
            ("decisions", self.MAX_DECISIONS),
        ):
            values = self.data.setdefault(key, [])
            if len(values) > limit:
                self.data[key] = values[-limit:]
        glossary = self.data.setdefault("glossary", {})
        if len(glossary) > self.MAX_GLOSSARY:
            keys = list(glossary)[: self.MAX_GLOSSARY]
            self.data["glossary"] = {k: glossary[k] for k in keys}
        chars = self.data.setdefault("characters", {})
        if len(chars) > self.MAX_CHARACTERS:
            keys = list(chars)[: self.MAX_CHARACTERS]
            self.data["characters"] = {k: chars[k] for k in keys}
        if save:
            self.save()

    def character_context(self, character_id: str) -> str:
        cid = str(character_id or "").strip()
        card = self.data.get("characters", {}).get(cid, {})
        if not card:
            return f"character_id={cid or 'unknown'}; no confirmed character facts"
        return (
            f"character_id={cid}; name={card.get('name', cid)}; "
            f"gender={card.get('gender', 'ambiguous')}; "
            f"age_group={card.get('age_group', 'unknown')}; "
            f"voice_profile={card.get('voice_profile', '')}; "
            f"personality={card.get('personality', '')}; "
            f"locked_voice={self.data.get('voice_locks', {}).get(cid, '')}"
        )[:1200]

    def context_for_segments(self, segments: list[dict]) -> str:
        ids = []
        for segment in segments:
            cid = str(segment.get("character_id") or "")
            if cid and cid not in ids:
                ids.append(cid)
        cards = [self.character_context(cid) for cid in ids[:20]]
        scenes = self.data.get("scene_summaries", [])[-4:]
        glossary = self.data.get("glossary", {})
        decisions = self.data.get("decisions", [])[-8:]
        parts = [
            "CANONICAL MOVIE BRAIN. Project state is the source of truth; never invent missing facts.",
            "Characters:\n" + "\n".join("- " + c for c in cards),
        ]
        if scenes:
            parts.append(
                "Recent scene summaries:\n" +
                "\n".join(
                    f"- {s.get('start', 0):.1f}-{s.get('end', 0):.1f}s: {s.get('summary', '')}"
                    for s in scenes
                )
            )
        if glossary:
            parts.append(
                "Glossary:\n" +
                "\n".join(f"- {k}: {v}" for k, v in list(glossary.items())[:30])
            )
        if decisions:
            parts.append(
                "Locked decisions:\n" + "\n".join(f"- {d}" for d in decisions)
            )
        return "\n\n".join(parts)[: self.MAX_CONTEXT_CHARS]

    def seed_prompt(self) -> str:
        return (
            "You are the continuity memory for a movie-dubbing pipeline. "
            "Treat the following compact canonical state as authoritative. "
            "Do not invent character facts, relationships, voices or events. "
            "Use unknown/ambiguous when evidence is insufficient. "
            "This is a read-only context seed for parallel downstream tasks.\n\n"
            + self.context_for_segments(
                [{"character_id": cid} for cid in list(self.data.get("characters", {}))[:20]]
            )
        )
