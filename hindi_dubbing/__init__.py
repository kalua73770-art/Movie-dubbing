"""
Hindi Dubbing Pipeline - Main package.
"""
from .config import Config, load_config, get_config, set_config
from .data.models import (
    ProjectData,
    DubSegment,
    SpeakerProfile,
    SpeakerSegment,
    TimeRange,
    SpeakerGender,
    SegmentStatus,
    QCFlag,
)
from .pipeline.orchestrator import Pipeline, create_pipeline

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_config",
    "get_config",
    "set_config",
    "ProjectData",
    "DubSegment",
    "SpeakerProfile",
    "SpeakerSegment",
    "TimeRange",
    "SpeakerGender",
    "SegmentStatus",
    "QCFlag",
    "Pipeline",
    "create_pipeline",
]