from __future__ import annotations

import logging

from groq import Groq


class GroqTranslator:
    """Drop-in duration-aware translator matching the former Gemini translation path."""

    def __init__(self, settings, log=None):
        self.s = settings
        self.log = log or logging.getLogger("groq.translation")
        if not self.s.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing or empty")
        self.client = Groq(
            api_key=self.s.groq_api_key,
            timeout=max(1.0, self.s.groq_timeout_ms / 1000.0),
            max_retries=1,
        )

    def translate_for_duration(
        self,
        segments,
        preferred_key_index: int | None = None,
        preferred_model_index: int | None = None,
        context_text: str = "",
        previous_interaction_id: str | None = None,
    ):
        # Keep the existing orchestrator signature/batching and MovieBrain hand-off.
        del preferred_key_index, preferred_model_index, previous_interaction_id

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
            "Write Hindi words in Devanagari script, not Roman/Hinglish transliteration, so the TTS engine "
            "uses natural Indian-Hindi pronunciation. Transliterate ordinary English words used as Hindi speech "
            "(for example detective, doctor, sir) into Devanagari when they are not proper names or acronyms. "
            "Keep proper names, product names and acronyms such as USR in their natural form. "
            "Aim for the requested spoken duration and word-count band. "
            "For longer lines, use natural Hindi phrasing, connectives or a brief natural reaction when "
            "needed, but never add unrelated information. For short lines, stay concise. "
            "Return exactly one line per input using ID|||Hindi text.\n\n"
            + "\n".join(lines)
        )

        messages = []
        if context_text.strip():
            messages.append({
                "role": "system",
                "content": "MovieBrain context for continuity (do not invent facts):\n" + context_text,
            })
        messages.append({"role": "user", "content": prompt})

        self.log.info(
            "Groq translation model=%s segments=%d",
            self.s.groq_translation_model,
            len(segments),
        )
        response = self.client.chat.completions.create(
            model=self.s.groq_translation_model,
            messages=messages,
            reasoning_effort="low",
        )
        output = (response.choices[0].message.content or "").strip()
        if not output:
            raise RuntimeError("Groq translation returned empty response")

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
            len(segments),
            self.s.groq_translation_model,
        )
        return result
