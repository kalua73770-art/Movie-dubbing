from __future__ import annotations

import logging

from hindi_dubbing.groq_translation import GroqTranslator
from hindi_dubbing.settings import settings


def main() -> None:
    translator = GroqTranslator(settings, logging.getLogger("groq-smoke"))
    segments = [{
        "id": "smoke_000001",
        "start": 0.0,
        "end": 2.2,
        "pace": "normal",
        "text": "Please tell me what happened here.",
    }]
    result = translator.translate_for_duration(
        segments,
        context_text="One male speaker; keep the meaning and a natural Hindi movie-dub tone.",
    )
    value = result.get("smoke_000001", "").strip()
    if not value:
        raise RuntimeError("Groq translation smoke test returned empty text")
    print("Groq translation smoke test OK:", value)
    print("Model:", settings.groq_translation_model)


if __name__ == "__main__":
    main()
