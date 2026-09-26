"""
Quality control and automated validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import librosa
from loguru import logger

from hindi_dubbing.data.models import (
    DubSegment, ProjectData, QCFlag, SegmentStatus, TimeRange
)
from hindi_dubbing.config import get_config


class QualityController:
    """Automated quality control for dubbing segments."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.qc_config = self.config.qc
    
    def check_segment(self, segment: DubSegment, project: ProjectData) -> list[QCFlag]:
        """Run all QC checks on a segment."""
        flags = []
        
        # 1. Duration mismatch
        if segment.adjusted_audio_path and segment.adjusted_audio_path.exists():
            audio, sr = librosa.load(str(segment.adjusted_audio_path), sr=None)
            actual_duration = len(audio) / sr
            target_duration = segment.time_range.duration
            diff = abs(actual_duration - target_duration)
            
            if diff > self.qc_config.max_duration_mismatch:
                flags.append(QCFlag.DURATION_MISMATCH)
        
        # 2. Speaker uncertainty (low confidence)
        if segment.confidence < self.qc_config.min_confidence:
            flags.append(QCFlag.SPEAKER_UNCERTAIN)
        
        # 3. Voice generation failed
        if segment.status == SegmentStatus.FAILED:
            flags.append(QCFlag.VOICE_GENERATION_FAILED)
        
        # 4. Audio overlap with adjacent segments
        flags.extend(self._check_overlap(segment, project))
        
        # 5. Timing mismatch (start time alignment)
        flags.extend(self._check_timing(segment))
        
        # 6. Translation length issues
        if segment.duration_ratio > self.config.translation.max_duration_ratio:
            flags.append(QCFlag.TRANSLATION_TOO_LONG)
        elif segment.duration_ratio < self.config.translation.min_duration_ratio:
            flags.append(QCFlag.TRANSLATION_TOO_SHORT)
        
        # 7. Unexpected silence in generated audio
        if segment.adjusted_audio_path and segment.adjusted_audio_path.exists():
            if self._has_unexpected_silence(segment.adjusted_audio_path):
                flags.append(QCFlag.UNEXPECTED_SILENCE)
        
        return flags
    
    def _check_overlap(self, segment: DubSegment, project: ProjectData) -> list[QCFlag]:
        """Check for overlap with adjacent segments."""
        flags = []
        
        for other in project.segments:
            if other.segment_id == segment.segment_id:
                continue
            if other.speaker_id != segment.speaker_id:
                continue  # Only check same speaker overlap
            
            if segment.time_range.overlaps(other.time_range):
                overlap = min(segment.time_range.end, other.time_range.end) - \
                         max(segment.time_range.start, other.time_range.start)
                if overlap > self.qc_config.max_overlap:
                    flags.append(QCFlag.AUDIO_OVERLAP)
                    break
        
        return flags
    
    def _check_timing(self, segment: DubSegment) -> list[QCFlag]:
        """Check timing alignment."""
        flags = []
        
        # Check if segment has reasonable timing
        if segment.time_range.duration < 0.1:
            flags.append(QCFlag.TIMING_MISMATCH)
        
        return flags
    
    def _has_unexpected_silence(self, audio_path: Path, threshold_db: float = -40) -> bool:
        """Check for unexpected silence in generated audio."""
        try:
            audio, sr = librosa.load(str(audio_path), sr=None)
            
            # Compute RMS in frames
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            rms_db = 20 * np.log10(rms + 1e-10)
            
            # Check for long silent regions
            silent = rms_db < threshold_db
            silent_frames = np.sum(silent)
            silent_duration = silent_frames * hop_length / sr
            
            return silent_duration > self.qc_config.flag_silence_threshold
        except Exception:
            return False
    
    def run_qc(self, project: ProjectData) -> ProjectData:
        """Run QC on all segments and generate summary."""
        logger.info(f"Running QC on {len(project.segments)} segments")
        
        flagged_count = 0
        flag_counts: dict[QCFlag, int] = {}
        
        for segment in project.segments:
            if segment.status in [SegmentStatus.COMPLETED, SegmentStatus.MIXED]:
                flags = self.check_segment(segment, project)
                segment.qc_flags = flags
                
                if flags:
                    flagged_count += 1
                    segment.status = SegmentStatus.FLAGGED
                    for flag in flags:
                        flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        # Generate summary
        project.qc_summary = {
            "total_segments": len(project.segments),
            "flagged_segments": flagged_count,
            "clean_segments": len(project.segments) - flagged_count,
            "flag_counts": {f.value: c for f, c in flag_counts.items()},
        }
        
        logger.info(f"QC complete: {flagged_count}/{len(project.segments)} segments flagged")
        for flag, count in flag_counts.items():
            logger.warning(f"  {flag.value}: {count}")
        
        return project
    
    def get_flagged_segments(self, project: ProjectData) -> list[DubSegment]:
        """Get all segments with QC flags."""
        return [s for s in project.segments if s.qc_flags]
    
    def generate_qc_report(self, project: ProjectData, output_path: Path) -> Path:
        """Generate human-readable QC report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Quality Control Report\n\n")
            f.write(f"Project: {project.project_id}\n")
            f.write(f"Total Segments: {project.qc_summary.get('total_segments', 0)}\n")
            f.write(f"Flagged: {project.qc_summary.get('flagged_segments', 0)}\n")
            f.write(f"Clean: {project.qc_summary.get('clean_segments', 0)}\n\n")
            
            f.write("## Flag Counts\n")
            for flag, count in project.qc_summary.get("flag_counts", {}).items():
                f.write(f"- {flag}: {count}\n")
            
            f.write("\n## Flagged Segments\n")
            for segment in self.get_flagged_segments(project):
                f.write(f"\n### Segment {segment.segment_id}\n")
                f.write(f"- Speaker: {segment.speaker_id}\n")
                f.write(f"- Character: {segment.character_name or 'N/A'}\n")
                f.write(f"- Time: {segment.time_range.start:.2f}s - {segment.time_range.end:.2f}s\n")
                f.write(f"- Original: {segment.original_text[:100]}...\n")
                f.write(f"- Hindi: {segment.hindi_text_adapted[:100]}...\n")
                f.write(f"- Duration Ratio: {segment.duration_ratio:.2f}\n")
                f.write(f"- Flags: {', '.join(f.value for f in segment.qc_flags)}\n")
        
        logger.info(f"QC report saved: {output_path}")
        return output_path


def process_qc(project: ProjectData, config: Optional[dict] = None) -> ProjectData:
    """Process quality control for project."""
    cfg = config or get_config()
    
    qc = QualityController(cfg)
    project = qc.run_qc(project)
    
    # Generate report
    report_dir = Path(cfg.paths.output_dir) / project.project_id
    report_dir.mkdir(parents=True, exist_ok=True)
    qc.generate_qc_report(project, report_dir / "qc_report.md")
    
    return project