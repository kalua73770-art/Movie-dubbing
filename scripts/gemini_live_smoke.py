from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from hindi_dubbing.gemini import GeminiService
from hindi_dubbing.settings import settings


def main() -> None:
    settings.validate()
    service = GeminiService(settings)

    text, text_model = service.generate_text(
        "Reply with exactly OK and nothing else."
    )
    if text.strip().upper() != "OK":
        raise RuntimeError(f"Unexpected text smoke response from {text_model}: {text!r}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        speech_wav = tmp_path / "speech.wav"
        subprocess.run(
            [
                "espeak-ng",
                "-w",
                str(speech_wav),
                "This is a short movie dubbing transcription test.",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        segments = service.transcribe(speech_wav)
        if not segments:
            raise RuntimeError("Live transcription smoke test returned no segments")

        tts_out = tmp_path / "tts.wav"
        tts_model = service.tts_segment(
            {
                "id": "smoke",
                "hindi": "यह एक छोटा परीक्षण है।",
                "emotion": "neutral",
                "pace": "normal",
            },
            settings.voices[0],
            tts_out,
        )
        if not tts_out.exists() or tts_out.stat().st_size < 100:
            raise RuntimeError(f"TTS smoke test created no usable audio: {tts_out}")

    print(f"LIVE_SMOKE_OK text_model={text_model} tts_model={tts_model} segments={len(segments)}")


if __name__ == "__main__":
    main()
