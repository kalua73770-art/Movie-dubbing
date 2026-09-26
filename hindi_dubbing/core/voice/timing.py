"""
Timing adjustment for generated audio to match original segments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import librosa
import soundfile as sf
from loguru import logger

from hindi_dubbing.data.models import DubSegment, TimeRange, SegmentStatus
from hindi_dubbing.config import get_config


class TimingAdjuster:
    """Adjust generated audio timing to match original segment timing."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.target_sample_rate = self.config.audio.sample_rate
    
    def adjust_segment(
        self,
        segment: DubSegment,
        output_path: Path,
    ) -> Path:
        """
        Adjust generated audio to match target time range.
        
        Strategy:
        1. If audio is close to target duration (±10%), time-stretch
        2. If audio is too long, speed up (with pitch preservation)
        3. If audio is too short, slow down or add silence padding
        4. Always align start time
        """
        if not segment.generated_audio_path or not segment.generated_audio_path.exists():
            logger.warning(f"No generated audio for {segment.segment_id}")
            segment.status = SegmentStatus.FAILED
            return output_path
        
        # Load generated audio
        audio, sr = librosa.load(str(segment.generated_audio_path), sr=self.target_sample_rate)
        generated_duration = len(audio) / sr
        target_duration = segment.time_range.duration
        
        if target_duration <= 0:
            logger.warning(f"Invalid target duration for {segment.segment_id}")
            segment.status = SegmentStatus.FAILED
            return output_path
        
        ratio = generated_duration / target_duration
        
        # Determine adjustment strategy
        if 0.9 <= ratio <= 1.1:
            # Close enough - time stretch
            adjusted = self._time_stretch(audio, sr, 1/ratio)
        elif ratio > 1.1:
            # Too long - speed up
            adjusted = self._time_stretch(audio, sr, 1/ratio)
        else:
            # Too short - slow down or pad
            if ratio > 0.5:
                adjusted = self._time_stretch(audio, sr, 1/ratio)
            else:
                # Very short - pad with silence
                adjusted = self._pad_audio(audio, target_duration, sr)
        
        # Ensure exact target length
        target_samples = int(target_duration * sr)
        if len(adjusted) > target_samples:
            adjusted = adjusted[:target_samples]
        elif len(adjusted) < target_samples:
            adjusted = np.pad(adjusted, (0, target_samples - len(adjusted)))
        
        # Save adjusted audio
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), adjusted, sr)
        
        segment.adjusted_audio_path = output_path
        segment.status = SegmentStatus.TIMING_ADJUSTED
        
        logger.debug(f"Adjusted {segment.segment_id}: {generated_duration:.2f}s -> {target_duration:.2f}s (ratio: {ratio:.2f})")
        
        return output_path
    
    def _time_stretch(self, audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
        """Time stretch audio preserving pitch."""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def _pad_audio(self, audio: np.ndarray, target_duration: float, sr: int) -> np.ndarray:
        """Pad audio with silence to target duration."""
        target_samples = int(target_duration * sr)
        if len(audio) >= target_samples:
            return audio[:target_samples]
        return np.pad(audio, (0, target_samples - len(audio)))


def process_timing_adjustment(project: ProjectData, config: Optional[dict] = None) -> ProjectData:
    """Process timing adjustment for all generated segments."""
    cfg = config or get_config()
    
    adjuster = TimingAdjuster(cfg)
    
    output_dir = Path(cfg.paths.temp_dir) / project.project_id / "adjusted"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Adjusting timing for {len(project.segments)} segments")
    
    for segment in project.segments:
        if segment.status != SegmentStatus.VOICE_GENERATED:
            continue
        
        output_path = output_dir / f"{segment.segment_id}.wav"
        adjuster.adjust_segment(segment, output_path)
    
    logger.success("Timing adjustment complete")
    return project