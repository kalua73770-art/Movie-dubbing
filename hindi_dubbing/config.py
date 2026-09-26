"""
Configuration management.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"
    extraction_codec: str = "pcm_s16le"


class DiarizationConfig(BaseModel):
    model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int = 1
    max_speakers: int = 20
    min_duration: float = 0.5


class TranscriptionConfig(BaseModel):
    model: str = "large-v3"
    language: str = "en"
    task: str = "transcribe"
    beam_size: int = 5
    temperature: float = 0.0
    condition_on_previous_text: bool = True
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


class TranslationConfig(BaseModel):
    model: str = "facebook/nllb-200-distilled-600M"
    source_lang: str = "eng_Latn"
    target_lang: str = "hin_Deva"
    max_length: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    max_duration_ratio: float = 1.15
    min_duration_ratio: float = 0.85
    rewrite_attempts: int = 3


class VoiceConfig(BaseModel):
    main_character_threshold: float = 0.05
    max_main_characters: int = 20
    tts_model: str = "coqui/XTTS-v2"
    tts_language: str = "hi"
    speaker_wav_duration: float = 6.0
    generic_voices: dict = {
        "male": ["generic_male_1", "generic_male_2", "generic_male_3"],
        "female": ["generic_female_1", "generic_female_2", "generic_female_3"],
    }
    pitch_variance: float = 0.1
    speed_variance: float = 0.1


class MixingConfig(BaseModel):
    separate_sources: bool = True
    separation_model: str = "htdemucs"
    dialogue_stem: str = "vocals"
    background_stems: list = ["drums", "bass", "other"]
    dialogue_level: float = 0.0
    bgm_level: float = -6.0
    sfx_level: float = -3.0
    ambience_level: float = -9.0
    normalize: bool = True
    target_lufs: float = -16.0
    apply_compression: bool = True
    compression_ratio: float = 3.0
    compression_threshold: float = -18.0


class QCConfig(BaseModel):
    max_duration_mismatch: float = 1.0
    min_confidence: float = 0.7
    max_overlap: float = 0.1
    min_gap: float = 0.05
    flag_silence_threshold: float = 0.5


class RenderConfig(BaseModel):
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    preset: str = "medium"
    crf: int = 20


class PathsConfig(BaseModel):
    cache_dir: str = ".cache"
    models_dir: str = "models"
    temp_dir: str = "temp"
    output_dir: str = "output"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


class ProjectConfig(BaseModel):
    name: str = "hindi_dubbing"
    version: str = "0.1.0"
    chunk_duration_seconds: int = 600
    chunk_overlap_seconds: int = 30


class Config(BaseModel):
    project: ProjectConfig = ProjectConfig()
    audio: AudioConfig = AudioConfig()
    diarization: DiarizationConfig = DiarizationConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    translation: TranslationConfig = TranslationConfig()
    voice: VoiceConfig = VoiceConfig()
    mixing: MixingConfig = MixingConfig()
    qc: QCConfig = QCConfig()
    render: RenderConfig = RenderConfig()
    paths: PathsConfig = PathsConfig()
    logging: LoggingConfig = LoggingConfig()


_config: Optional[Config] = None


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file."""
    global _config
    
    if _config is not None:
        return _config
    
    # Default config
    config = Config()
    
    # Override with file if provided
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f)
        
        # Deep merge
        def merge_dict(base: dict, override: dict) -> dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        config_dict = config.model_dump()
        merged = merge_dict(config_dict, file_config)
        config = Config(**merged)
    
    _config = config
    return _config


def get_config() -> Config:
    """Get current configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration."""
    global _config
    _config = config