from dataclasses import dataclass, field
import os


def _csv(name, default):
    value = os.getenv(name, default)
    return [x.strip() for x in value.split(",") if x.strip()]


def _bool(name, default):
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    api_keys: list[str] = field(default_factory=lambda: _csv("GEMINI_API_KEYS", ""))

    # Task-specific pools. Do not mix transcription/TTS models with normal text models.
    text_models: list[str] = field(
        default_factory=lambda: _csv(
            "GEMINI_TEXT_MODELS",
            "gemini-3.5-flash-lite,gemini-3.1-flash-lite",
        )
    )
    tts_models: list[str] = field(
        default_factory=lambda: _csv(
            "GEMINI_TTS_MODELS",
            "gemini-3.8-flash-tts,gemini-3.8-flash-lite-tts",
        )
    )
    transcribe_model: str = os.getenv(
        "GEMINI_TRANSCRIBE_MODEL",
        "gemini-3.5-transcribe",
    )
    video_models: list[str] = field(
        default_factory=lambda: _csv(
            "GEMINI_VIDEO_MODELS",
            "gemini-3.6-flash,gemini-3.5-flash-lite",
        )
    )

    # Keep transcription comfortably below the 30-minute diarization/timestamps limit.
    transcribe_chunk_seconds: int = int(os.getenv("TRANSCRIBE_CHUNK_SECONDS", "900"))
    transcribe_overlap_seconds: int = int(os.getenv("TRANSCRIBE_OVERLAP_SECONDS", "8"))
    transcribe_workers: int = int(os.getenv("TRANSCRIBE_WORKERS", "4"))
    translation_workers: int = int(os.getenv("TRANSLATION_WORKERS", "4"))
    tts_workers: int = int(os.getenv("TTS_WORKERS", "4"))

    # Video analysis is the most failure-prone/slowest optional stage.
    video_analysis_window_seconds: int = int(
        os.getenv("VIDEO_ANALYSIS_WINDOW_SECONDS", "180")
    )
    video_analysis_max_windows: int = int(
        os.getenv("VIDEO_ANALYSIS_MAX_WINDOWS", "6")
    )
    video_max_keys_per_model: int = int(
        os.getenv("VIDEO_MAX_KEYS_PER_MODEL", "2")
    )
    enable_video_analysis: bool = _bool("ENABLE_VIDEO_ANALYSIS", True)

    # Network timeouts are intentionally task-specific.
    gemini_http_timeout_ms: int = int(
        os.getenv("GEMINI_HTTP_TIMEOUT_MS", "600000")
    )
    text_timeout_ms: int = int(os.getenv("GEMINI_TEXT_TIMEOUT_MS", "45000"))
    transcribe_timeout_ms: int = int(
        os.getenv("GEMINI_TRANSCRIBE_TIMEOUT_MS", "600000")
    )
    tts_timeout_ms: int = int(os.getenv("GEMINI_TTS_TIMEOUT_MS", "60000"))
    video_timeout_ms: int = int(os.getenv("GEMINI_VIDEO_TIMEOUT_MS", "90000"))
    speaker_analysis_timeout_ms: int = int(
        os.getenv("GEMINI_SPEAKER_ANALYSIS_TIMEOUT_MS", "120000")
    )

    # The router remembers failures instead of hammering an exhausted/broken key.
    gemini_quota_cooldown_seconds: int = int(
        os.getenv("GEMINI_QUOTA_COOLDOWN_SECONDS", "300")
    )
    gemini_auth_cooldown_seconds: int = int(
        os.getenv("GEMINI_AUTH_COOLDOWN_SECONDS", "900")
    )
    gemini_transient_cooldown_seconds: int = int(
        os.getenv("GEMINI_TRANSIENT_COOLDOWN_SECONDS", "30")
    )

    translation_batch_chars: int = int(
        os.getenv("TRANSLATION_BATCH_CHARS", "12000")
    )
    translation_chars_per_second: float = float(
        os.getenv("TRANSLATION_CHARS_PER_SECOND", "12")
    )

    target_tts_min_ratio: float = float(
        os.getenv("TARGET_TTS_MIN_RATIO", "0.60")
    )
    target_tts_max_ratio: float = float(
        os.getenv("TARGET_TTS_MAX_RATIO", "1.40")
    )
    rewrite_attempts: int = int(os.getenv("TTS_REWRITE_ATTEMPTS", "1"))
    background_scene_threshold: float = float(os.getenv("BACKGROUND_SCENE_THRESHOLD", "0.35"))

    speaker_analysis_model: str = os.getenv(
        "GEMINI_SPEAKER_MODEL",
        "gemini-3.5-flash-lite",
    )

    audio_stem_provider: str = os.getenv("AUDIO_STEM_PROVIDER", "reconstruct")
    audio_separation_chunk_seconds: int = int(
        os.getenv("AUDIO_SEPARATION_CHUNK_SECONDS", "600")
    )
    audio_separation_space: str = os.getenv(
        "AUDIO_SEPARATION_SPACE",
        "WAVbot/TIGER-audio-extraction",
    )

    work_dir: str = os.getenv("WORK_DIR", "/tmp/movie-dubbing")
    output_dir: str = os.getenv(
        "OUTPUT_DIR",
        "/tmp/movie-dubbing/outputs",
    )

    voices: list[str] = field(
        default_factory=lambda: _csv(
            "GEMINI_VOICES",
            "Kore,Puck,Charon,Zephyr,Fenrir,Leda,Orus,Aoede",
        )
    )

    def validate(self):
        if not self.api_keys:
            raise RuntimeError("GEMINI_API_KEYS is missing or empty")
        if self.transcribe_chunk_seconds <= self.transcribe_overlap_seconds:
            raise RuntimeError(
                "TRANSCRIBE_CHUNK_SECONDS must be greater than "
                "TRANSCRIBE_OVERLAP_SECONDS"
            )
        if self.transcribe_workers < 1:
            raise RuntimeError("TRANSCRIBE_WORKERS must be >= 1")
        if self.translation_workers < 1:
            raise RuntimeError("TRANSLATION_WORKERS must be >= 1")
        if self.tts_workers < 1:
            raise RuntimeError("TTS_WORKERS must be >= 1")
        if self.video_analysis_max_windows < 1:
            raise RuntimeError("VIDEO_ANALYSIS_MAX_WINDOWS must be >= 1")
        for name, value in {
            "GEMINI_HTTP_TIMEOUT_MS": self.gemini_http_timeout_ms,
            "GEMINI_TEXT_TIMEOUT_MS": self.text_timeout_ms,
            "GEMINI_TRANSCRIBE_TIMEOUT_MS": self.transcribe_timeout_ms,
            "GEMINI_TTS_TIMEOUT_MS": self.tts_timeout_ms,
            "GEMINI_VIDEO_TIMEOUT_MS": self.video_timeout_ms,
        }.items():
            if value <= 0:
                raise RuntimeError(f"{name} must be > 0")


settings = Settings()


__all__ = ["Settings", "settings"]
