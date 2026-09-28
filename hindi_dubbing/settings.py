from dataclasses import dataclass, field
import os


def _csv(name, default):
    value = os.getenv(name, default)
    return [x.strip() for x in value.split(',') if x.strip()]


@dataclass
class Settings:
    api_keys: list[str] = field(default_factory=lambda: _csv('GEMINI_API_KEYS', ''))

    text_models: list[str] = field(
        default_factory=lambda: _csv(
            'GEMINI_TEXT_MODELS',
            'gemini-3.5-flash-lite,gemini-2.5-flash-lite',
        )
    )
    tts_models: list[str] = field(
        default_factory=lambda: _csv(
            'GEMINI_TTS_MODELS',
            'gemini-3.8-flash-tts,gemini-3.8-flash-lite-tts',
        )
    )
    transcribe_model: str = os.getenv('GEMINI_TRANSCRIBE_MODEL', 'gemini-3.5-transcribe')
    video_models: list[str] = field(
        default_factory=lambda: _csv(
            'GEMINI_VIDEO_MODELS',
            'gemini-3.8-flash,gemini-3.5-flash-lite,gemini-2.5-flash',
        )
    )

    transcribe_chunk_seconds: int = int(os.getenv('TRANSCRIBE_CHUNK_SECONDS', '1740'))
    transcribe_overlap_seconds: int = int(os.getenv('TRANSCRIBE_OVERLAP_SECONDS', '20'))
    video_analysis_window_seconds: int = int(os.getenv('VIDEO_ANALYSIS_WINDOW_SECONDS', '600'))
    gemini_http_timeout_ms: int = int(os.getenv('GEMINI_HTTP_TIMEOUT_MS', '150000'))

    translation_batch_chars: int = int(os.getenv('TRANSLATION_BATCH_CHARS', '6000'))
    translation_chars_per_second: float = float(os.getenv('TRANSLATION_CHARS_PER_SECOND', '12'))

    target_tts_min_ratio: float = float(os.getenv('TARGET_TTS_MIN_RATIO', '0.88'))
    target_tts_max_ratio: float = float(os.getenv('TARGET_TTS_MAX_RATIO', '1.12'))
    rewrite_attempts: int = int(os.getenv('TTS_REWRITE_ATTEMPTS', '2'))

    speaker_analysis_model: str = os.getenv('GEMINI_SPEAKER_MODEL', 'gemini-3.5-flash-lite')

    audio_stem_provider: str = os.getenv('AUDIO_STEM_PROVIDER', 'tiger_hf')
    audio_separation_chunk_seconds: int = int(os.getenv('AUDIO_SEPARATION_CHUNK_SECONDS', '600'))
    audio_separation_space: str = os.getenv('AUDIO_SEPARATION_SPACE', 'WAVbot/TIGER-audio-extraction')

    work_dir: str = os.getenv('WORK_DIR', '/tmp/movie-dubbing')
    output_dir: str = os.getenv('OUTPUT_DIR', '/tmp/movie-dubbing/outputs')

    voices: list[str] = field(
        default_factory=lambda: _csv(
            'GEMINI_VOICES',
            'Kore,Puck,Charon,Zephyr,Fenrir,Leda,Orus,Aoede',
        )
    )

    def validate(self):
        if not self.api_keys:
            raise RuntimeError('GEMINI_API_KEYS is missing or empty')


settings = Settings()


__all__ = ['Settings', 'settings']
