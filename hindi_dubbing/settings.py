from dataclasses import dataclass, field
import os


def _csv(name, default):
    value=os.getenv(name, default)
    return [x.strip() for x in value.split(",") if x.strip()]

@dataclass
class Settings:
    api_keys: list[str]=field(default_factory=lambda:_csv("GEMINI_API_KEYS",""))
    text_models: list[str]=field(default_factory=lambda:_csv("GEMINI_TEXT_MODELS","gemini-3.5-flash-lite,gemini-2.5-flash-lite"))
    tts_models: list[str]=field(default_factory=lambda:_csv("GEMINI_TTS_MODELS","gemini-2.5-flash-preview-tts,gemini-3.1-flash-tts-preview"))
    transcribe_model: str=os.getenv("GEMINI_TRANSCRIBE_MODEL","gemini-3.5-transcribe")
    transcribe_chunk_seconds: int=int(os.getenv("TRANSCRIBE_CHUNK_SECONDS","1740"))
    transcribe_overlap_seconds: int=int(os.getenv("TRANSCRIBE_OVERLAP_SECONDS","20"))
    translation_batch_chars: int=int(os.getenv("TRANSLATION_BATCH_CHARS","10000"))
    tts_batch_seconds: int=int(os.getenv("TTS_BATCH_SECONDS","600"))
    tts_batch_chars: int=int(os.getenv("TTS_BATCH_CHARS","8500"))
    tts_pause_tag: str=os.getenv("TTS_PAUSE_TAG","<short pause>")
    duck_db: float=float(os.getenv("ORIGINAL_DUCK_DB","-40"))
    speaker_analysis_model: str=os.getenv("GEMINI_SPEAKER_MODEL","gemini-3.5-flash-lite")
    work_dir: str=os.getenv("WORK_DIR","/tmp/movie-dubbing")
    output_dir: str=os.getenv("OUTPUT_DIR","/tmp/movie-dubbing/outputs")
    voices: list[str]=field(default_factory=lambda:_csv("GEMINI_VOICES","Kore,Puck,Charon,Zephyr,Fenrir,Leda,Orus,Aoede"))

    def validate(self):
        if not self.api_keys: raise RuntimeError("GEMINI_API_KEYS is missing or empty")

settings=Settings()
