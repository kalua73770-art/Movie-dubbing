"""
Data models and schemas for the Hindi Dubbing Pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Optional
import json


class SpeakerGender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class SegmentStatus(str, Enum):
    PENDING = "pending"
    TRANSCRIBED = "transcribed"
    TRANSLATED = "translated"
    VOICE_GENERATED = "voice_generated"
    TIMING_ADJUSTED = "timing_adjusted"
    MIXED = "mixed"
    COMPLETED = "completed"
    FLAGGED = "flagged"
    FAILED = "failed"


class QCFlag(str, Enum):
    DURATION_MISMATCH = "duration_mismatch"
    SPEAKER_UNCERTAIN = "speaker_uncertain"
    VOICE_GENERATION_FAILED = "voice_generation_failed"
    AUDIO_OVERLAP = "audio_overlap"
    TIMING_MISMATCH = "timing_mismatch"
    LOW_CONFIDENCE = "low_confidence"
    UNEXPECTED_SILENCE = "unexpected_silence"
    TRANSLATION_TOO_LONG = "translation_too_long"
    TRANSLATION_TOO_SHORT = "translation_too_short"


@dataclass
class TimeRange:
    """Represents a time range in seconds."""
    start: float
    end: float
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def overlaps(self, other: TimeRange) -> bool:
        return self.start < other.end and other.start < self.end
    
    def to_timedelta(self) -> tuple[timedelta, timedelta]:
        return (timedelta(seconds=self.start), timedelta(seconds=self.end))
    
    @classmethod
    def from_timedelta(cls, start: timedelta, end: timedelta) -> TimeRange:
        return cls(start.total_seconds(), end.total_seconds())
    
    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "duration": self.duration}
    
    @classmethod
    def from_dict(cls, data: dict) -> TimeRange:
        return cls(data["start"], data["end"])


@dataclass
class SpeakerSegment:
    """A single speech segment from diarization."""
    speaker_id: str
    time_range: TimeRange
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "time_range": self.time_range.to_dict(),
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> SpeakerSegment:
        return cls(
            speaker_id=data["speaker_id"],
            time_range=TimeRange.from_dict(data["time_range"]),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class DubSegment:
    """A complete dubbing unit with all metadata."""
    # Identification
    segment_id: str
    speaker_id: str
    character_name: Optional[str] = None
    
    # Timing
    time_range: TimeRange = field(default_factory=lambda: TimeRange(0, 0))
    
    # Original content
    original_text: str = ""
    original_language: str = "en"
    
    # Hindi adaptation
    hindi_text: str = ""
    hindi_text_adapted: str = ""  # Duration-matched version
    
    # Voice
    voice_id: str = ""
    voice_type: str = "main"  # "main" or "generic"
    gender: SpeakerGender = SpeakerGender.UNKNOWN
    
    # Generated audio
    generated_audio_path: Optional[Path] = None
    adjusted_audio_path: Optional[Path] = None
    final_audio_path: Optional[Path] = None
    
    # Metadata
    status: SegmentStatus = SegmentStatus.PENDING
    confidence: float = 1.0
    qc_flags: list[QCFlag] = field(default_factory=list)
    duration_ratio: float = 1.0  # hindi_duration / original_duration
    
    # Processing info
    rewrite_attempts: int = 0
    generation_params: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "speaker_id": self.speaker_id,
            "character_name": self.character_name,
            "time_range": self.time_range.to_dict(),
            "original_text": self.original_text,
            "original_language": self.original_language,
            "hindi_text": self.hindi_text,
            "hindi_text_adapted": self.hindi_text_adapted,
            "voice_id": self.voice_id,
            "voice_type": self.voice_type,
            "gender": self.gender.value,
            "generated_audio_path": str(self.generated_audio_path) if self.generated_audio_path else None,
            "adjusted_audio_path": str(self.adjusted_audio_path) if self.adjusted_audio_path else None,
            "final_audio_path": str(self.final_audio_path) if self.final_audio_path else None,
            "status": self.status.value,
            "confidence": self.confidence,
            "qc_flags": [f.value for f in self.qc_flags],
            "duration_ratio": self.duration_ratio,
            "rewrite_attempts": self.rewrite_attempts,
            "generation_params": self.generation_params,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> DubSegment:
        return cls(
            segment_id=data["segment_id"],
            speaker_id=data["speaker_id"],
            character_name=data.get("character_name"),
            time_range=TimeRange.from_dict(data["time_range"]),
            original_text=data.get("original_text", ""),
            original_language=data.get("original_language", "en"),
            hindi_text=data.get("hindi_text", ""),
            hindi_text_adapted=data.get("hindi_text_adapted", ""),
            voice_id=data.get("voice_id", ""),
            voice_type=data.get("voice_type", "main"),
            gender=SpeakerGender(data.get("gender", "unknown")),
            generated_audio_path=Path(data["generated_audio_path"]) if data.get("generated_audio_path") else None,
            adjusted_audio_path=Path(data["adjusted_audio_path"]) if data.get("adjusted_audio_path") else None,
            final_audio_path=Path(data["final_audio_path"]) if data.get("final_audio_path") else None,
            status=SegmentStatus(data.get("status", "pending")),
            confidence=data.get("confidence", 1.0),
            qc_flags=[QCFlag(f) for f in data.get("qc_flags", [])],
            duration_ratio=data.get("duration_ratio", 1.0),
            rewrite_attempts=data.get("rewrite_attempts", 0),
            generation_params=data.get("generation_params", {}),
        )


@dataclass
class SpeakerProfile:
    """Profile for a speaker/character."""
    speaker_id: str
    character_name: Optional[str] = None
    gender: SpeakerGender = SpeakerGender.UNKNOWN
    voice_id: str = ""
    voice_type: str = "main"  # "main" or "generic"
    total_speech_time: float = 0.0
    segment_count: int = 0
    frequency_percent: float = 0.0
    reference_audio_path: Optional[Path] = None
    embedding: Optional[list[float]] = None  # Speaker embedding for consistency
    
    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "character_name": self.character_name,
            "gender": self.gender.value,
            "voice_id": self.voice_id,
            "voice_type": self.voice_type,
            "total_speech_time": self.total_speech_time,
            "segment_count": self.segment_count,
            "frequency_percent": self.frequency_percent,
            "reference_audio_path": str(self.reference_audio_path) if self.reference_audio_path else None,
            "embedding": self.embedding,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> SpeakerProfile:
        return cls(
            speaker_id=data["speaker_id"],
            character_name=data.get("character_name"),
            gender=SpeakerGender(data.get("gender", "unknown")),
            voice_id=data.get("voice_id", ""),
            voice_type=data.get("voice_type", "main"),
            total_speech_time=data.get("total_speech_time", 0.0),
            segment_count=data.get("segment_count", 0),
            frequency_percent=data.get("frequency_percent", 0.0),
            reference_audio_path=Path(data["reference_audio_path"]) if data.get("reference_audio_path") else None,
            embedding=data.get("embedding"),
        )


@dataclass
class ProjectData:
    """Master project data containing all dubbing information."""
    project_id: str
    source_video_path: Path
    source_audio_path: Optional[Path] = None
    created_at: str = ""
    updated_at: str = ""
    
    # Audio info
    total_duration: float = 0.0
    sample_rate: int = 16000
    channels: int = 1
    
    # Speaker profiles
    speakers: dict[str, SpeakerProfile] = field(default_factory=dict)
    
    # Segments
    segments: list[DubSegment] = field(default_factory=list)
    
    # Processing state
    current_stage: str = "initialized"
    completed_stages: list[str] = field(default_factory=list)
    
    # QC summary
    qc_summary: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "source_video_path": str(self.source_video_path),
            "source_audio_path": str(self.source_audio_path) if self.source_audio_path else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_duration": self.total_duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "speakers": {k: v.to_dict() for k, v in self.speakers.items()},
            "segments": [s.to_dict() for s in self.segments],
            "current_stage": self.current_stage,
            "completed_stages": self.completed_stages,
            "qc_summary": self.qc_summary,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ProjectData:
        return cls(
            project_id=data["project_id"],
            source_video_path=Path(data["source_video_path"]),
            source_audio_path=Path(data["source_audio_path"]) if data.get("source_audio_path") else None,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            total_duration=data.get("total_duration", 0.0),
            sample_rate=data.get("sample_rate", 16000),
            channels=data.get("channels", 1),
            speakers={k: SpeakerProfile.from_dict(v) for k, v in data.get("speakers", {}).items()},
            segments=[DubSegment.from_dict(s) for s in data.get("segments", [])],
            current_stage=data.get("current_stage", "initialized"),
            completed_stages=data.get("completed_stages", []),
            qc_summary=data.get("qc_summary", {}),
        )
    
    def save(self, path: Path) -> None:
        """Save project data to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> ProjectData:
        """Load project data from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_segments_by_speaker(self, speaker_id: str) -> list[DubSegment]:
        """Get all segments for a specific speaker."""
        return [s for s in self.segments if s.speaker_id == speaker_id]
    
    def get_flagged_segments(self) -> list[DubSegment]:
        """Get all segments with QC flags."""
        return [s for s in self.segments if s.qc_flags]
    
    def get_segments_by_status(self, status: SegmentStatus) -> list[DubSegment]:
        """Get all segments with a specific status."""
        return [s for s in self.segments if s.status == status]
    
    def update_speaker_stats(self) -> None:
        """Recalculate speaker statistics."""
        total_time = sum(s.time_range.duration for s in self.segments)
        speaker_times: dict[str, float] = {}
        speaker_counts: dict[str, int] = {}
        
        for segment in self.segments:
            dur = segment.time_range.duration
            speaker_times[segment.speaker_id] = speaker_times.get(segment.speaker_id, 0) + dur
            speaker_counts[segment.speaker_id] = speaker_counts.get(segment.speaker_id, 0) + 1
        
        for speaker_id, profile in self.speakers.items():
            profile.total_speech_time = speaker_times.get(speaker_id, 0)
            profile.segment_count = speaker_counts.get(speaker_id, 0)
            profile.frequency_percent = (profile.total_speech_time / total_time * 100) if total_time > 0 else 0