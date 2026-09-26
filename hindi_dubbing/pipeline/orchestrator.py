"""
Pipeline orchestration - coordinates all stages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable
from datetime import datetime
import uuid
import shutil
import os
from loguru import logger

from hindi_dubbing.data.models import ProjectData, SegmentStatus
from hindi_dubbing.config import get_config, load_config
from hindi_dubbing.core.audio.processor import extract_audio, get_audio_info, split_audio_chunks
from hindi_dubbing.core.diarization.engine import DiarizationEngine, classify_speaker_gender
from hindi_dubbing.core.transcription.engine import TranscriptionEngine
from hindi_dubbing.core.translation.engine import process_translation
from hindi_dubbing.core.voice.engine import process_voice_generation
from hindi_dubbing.core.voice.timing import process_timing_adjustment
from hindi_dubbing.core.mixing.mixer import process_mixing
from hindi_dubbing.core.qc.engine import process_qc
from hindi_dubbing.core.render.renderer import render_final_video, render_preview


class PipelineStage:
    """Base class for pipeline stages."""
    
    name: str = "base"
    description: str = "Base stage"
    
    def run(self, project: ProjectData, config: dict) -> ProjectData:
        raise NotImplementedError


class Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = load_config(config_path)
        self.project: Optional[ProjectData] = None
        self.callbacks: list[Callable] = []
    
    def add_callback(self, callback: Callable) -> None:
        """Add progress callback."""
        self.callbacks.append(callback)
    
    def _notify(self, stage: str, message: str, progress: float = 0.0) -> None:
        """Notify callbacks of progress."""
        for cb in self.callbacks:
            try:
                cb(stage, message, progress)
            except Exception:
                pass
    
    def create_project(self, video_path: Path, project_dir: Path) -> ProjectData:
        """Create new project from video file."""
        project_id = uuid.uuid4().hex[:12]
        
        # Extract audio
        audio_path = project_dir / "source_audio.wav"
        extract_audio(video_path, audio_path, 
                     sample_rate=self.config.audio.sample_rate,
                     channels=self.config.audio.channels)
        
        # Get audio info
        info = get_audio_info(audio_path)
        
        # Create project
        project = ProjectData(
            project_id=project_id,
            source_video_path=video_path,
            source_audio_path=audio_path,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            total_duration=info["duration"],
            sample_rate=info["sample_rate"],
            channels=info["channels"],
            current_stage="initialized",
        )
        
        # Save initial project
        project.save(project_dir / "project.json")
        
        self.project = project
        return project
    
    def load_project(self, project_dir: Path) -> ProjectData:
        """Load existing project."""
        project_path = project_dir / "project.json"
        if not project_path.exists():
            raise FileNotFoundError(f"Project not found: {project_path}")
        
        project = ProjectData.load(project_path)
        self.project = project
        return project
    
    def save_project(self) -> None:
        """Save current project."""
        if self.project:
            project_dir = self.project.source_video_path.parent
            self.project.updated_at = datetime.now().isoformat()
            self.project.save(project_dir / "project.json")
    
    def run_stage(self, stage_name: str, stage_func: Callable, *args, **kwargs) -> ProjectData:
        """Run a pipeline stage with error handling."""
        if not self.project:
            raise RuntimeError("No project loaded")
        
        self._notify(stage_name, f"Starting {stage_name}", 0.0)
        logger.info(f"=== Stage: {stage_name} ===")
        
        try:
            self.project = stage_func(self.project, self.config, *args, **kwargs)
            self.project.current_stage = stage_name
            self.project.completed_stages.append(stage_name)
            self.project.updated_at = datetime.now().isoformat()
            self.save_project()
            
            self._notify(stage_name, f"Completed {stage_name}", 1.0)
            logger.success(f"Stage {stage_name} completed")
            
        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            self._notify(stage_name, f"Failed: {e}", 0.0)
            raise
        
        return self.project
    
    def run_full_pipeline(self, video_path: Path, output_path: Path) -> ProjectData:
        """Run complete dubbing pipeline."""
        project_dir = output_path.parent / f"project_{output_path.stem}"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage 1: Create project & extract audio
        self.create_project(video_path, project_dir)
        
        # Stage 2: Audio analysis (diarization + transcription)
        self.run_stage("audio_analysis", self._stage_audio_analysis)
        
        # Stage 3: Translation
        self.run_stage("translation", process_translation)
        
        # Stage 4: Voice generation
        self.run_stage("voice_generation", process_voice_generation)
        
        # Stage 5: Timing adjustment
        self.run_stage("timing_adjustment", process_timing_adjustment)
        
        # Stage 6: Mixing
        self.run_stage("mixing", process_mixing)
        
        # Stage 7: QC
        self.run_stage("qc", process_qc)
        
        # Stage 8: Final render
        self.run_stage("final_render", self._stage_final_render, output_path)
        
        return self.project
    
    def _stage_audio_analysis(self, project: ProjectData, config: dict) -> ProjectData:
        """Stage 1: Audio analysis - diarization and transcription."""
        # Split audio into chunks
        chunk_dir = Path(config.paths.temp_dir) / project.project_id / "chunks"
        chunks = split_audio_chunks(
            project.source_audio_path,
            config.project.chunk_duration_seconds,
            config.project.chunk_overlap_seconds,
            chunk_dir,
        )
        
        # Diarization
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        diarization = DiarizationEngine(
            model_name=config.diarization.model,
            min_speakers=config.diarization.min_speakers,
            max_speakers=config.diarization.max_speakers,
            hf_token=hf_token,
        )
        
        segments = diarization.diarize_chunks(
            [c["path"] for c in chunks],
            chunks,
        )
        
        # Classify gender
        gender_map = classify_speaker_gender(project.source_audio_path, segments)
        
        # Create speaker profiles
        for segment in segments:
            speaker_id = segment.speaker_id
            if speaker_id not in project.speakers:
                project.speakers[speaker_id] = type(project.speakers[speaker_id])(
                    speaker_id=speaker_id,
                    gender=gender_map.get(speaker_id, "unknown"),
                )
        
        # Create dub segments
        project.segments = []
        for i, seg in enumerate(segments):
            dub_seg = type(project.segments[0])(
                segment_id=f"seg_{i:06d}",
                speaker_id=seg.speaker_id,
                time_range=seg.time_range,
                confidence=seg.confidence,
            )
            project.segments.append(dub_seg)
        
        # Transcription
        transcription = TranscriptionEngine(
            model_name=config.transcription.model,
            language=config.transcription.language,
            beam_size=config.transcription.beam_size,
            temperature=config.transcription.temperature,
        )
        
        project.segments = transcription.transcribe(project.source_audio_path, segments)
        
        return project
    
    def _stage_final_render(self, project: ProjectData, config: dict, output_path: Path) -> ProjectData:
        """Stage 8: Final video render."""
        render_final_video(project, output_path, config)
        return project
    
    # Individual stage methods for granular control
    def run_audio_analysis(self) -> ProjectData:
        return self.run_stage("audio_analysis", self._stage_audio_analysis)
    
    def run_translation(self) -> ProjectData:
        return self.run_stage("translation", process_translation)
    
    def run_voice_generation(self) -> ProjectData:
        return self.run_stage("voice_generation", process_voice_generation)
    
    def run_timing_adjustment(self) -> ProjectData:
        return self.run_stage("timing_adjustment", process_timing_adjustment)
    
    def run_mixing(self) -> ProjectData:
        return self.run_stage("mixing", process_mixing)
    
    def run_qc(self) -> ProjectData:
        return self.run_stage("qc", process_qc)
    
    def run_final_render(self, output_path: Path) -> ProjectData:
        return self.run_stage("final_render", self._stage_final_render, output_path)
    
    def render_preview(self, output_path: Path, start: float = 0, duration: float = 60) -> Path:
        """Render a preview clip."""
        if not self.project:
            raise RuntimeError("No project loaded")
        return render_preview(self.project, output_path, start, duration, self.config)


def create_pipeline(config_path: Optional[Path] = None) -> Pipeline:
    """Factory function to create pipeline."""
    return Pipeline(config_path)