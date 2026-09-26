"""
Audio mixing: combine Hindi dialogue with original BGM/SFX.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import numpy as np
import librosa
import soundfile as sf
import sys
import subprocess
from loguru import logger

from hindi_dubbing.data.models import DubSegment, ProjectData, SegmentStatus
from hindi_dubbing.config import get_config
from hindi_dubbing.core.audio.processor import normalize_audio, apply_compression


class AudioMixer:
    """Mix Hindi dialogue with original background audio."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.mix_config = self.config.mixing
        self.target_sr = self.config.audio.sample_rate
    
    def separate_sources(self, audio_path: Path, output_dir: Path) -> dict:
        """
        Separate audio into stems using Demucs.
        
        Returns:
            Dict with stem paths: vocals, drums, bass, other
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-o", str(output_dir),
            "--two-stems=vocals",
            str(audio_path),
        ]
        
        logger.info(f"Separating sources for {audio_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Demucs failed: {result.stderr}")
            raise RuntimeError(f"Source separation failed: {result.stderr}")
        
        # Find output files
        stem_dir = output_dir / "htdemucs" / audio_path.stem
        stems = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            stem_path = stem_dir / f"{stem}.wav"
            if stem_path.exists():
                stems[stem] = stem_path
        
        logger.success(f"Separated {len(stems)} stems")
        return stems
    
    def create_dialogue_track(
        self,
        project: ProjectData,
        total_duration: float,
        output_path: Path,
    ) -> Path:
        """
        Create continuous dialogue track from all segments.
        Places each segment at its correct timestamp.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_samples = int(total_duration * self.target_sr)
        dialogue_track = np.zeros(total_samples, dtype=np.float32)
        
        for segment in project.segments:
            if segment.status != SegmentStatus.TIMING_ADJUSTED:
                continue
            
            if not segment.adjusted_audio_path or not segment.adjusted_audio_path.exists():
                continue
            
            # Load segment audio
            audio, sr = librosa.load(str(segment.adjusted_audio_path), sr=self.target_sr)
            
            # Calculate position
            start_sample = int(segment.time_range.start * self.target_sr)
            end_sample = start_sample + len(audio)
            
            # Handle overlap (crossfade)
            if end_sample > total_samples:
                audio = audio[:total_samples - start_sample]
                end_sample = total_samples
            
            if start_sample < total_samples:
                # Simple overlay (later segments overwrite earlier)
                # In practice, segments shouldn't overlap if diarization is correct
                dialogue_track[start_sample:end_sample] = audio
        
        sf.write(str(output_path), dialogue_track, self.target_sr)
        logger.info(f"Created dialogue track: {output_path}")
        return output_path
    
    def mix_tracks(
        self,
        dialogue_path: Path,
        background_stems: dict,
        output_path: Path,
    ) -> Path:
        """
        Mix dialogue with background stems.
        
        Args:
            dialogue_path: Path to dialogue track
            background_stems: Dict of stem_name -> path
            output_path: Output path for final mix
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load dialogue
        dialogue, sr = librosa.load(str(dialogue_path), sr=self.target_sr)
        
        # Load and mix background stems
        background = np.zeros_like(dialogue)
        
        stem_levels = {
            "drums": self.mix_config.bgm_level,
            "bass": self.mix_config.bgm_level,
            "other": self.mix_config.bgm_level,
        }
        
        for stem_name, stem_path in background_stems.items():
            if stem_name == "vocals":
                continue  # Skip original vocals
            
            stem_audio, _ = librosa.load(str(stem_path), sr=self.target_sr)
            
            # Match length
            if len(stem_audio) > len(background):
                stem_audio = stem_audio[:len(background)]
            elif len(stem_audio) < len(background):
                stem_audio = np.pad(stem_audio, (0, len(background) - len(stem_audio)))
            
            # Apply level
            level_db = stem_levels.get(stem_name, self.mix_config.bgm_level)
            level_linear = 10 ** (level_db / 20)
            background += stem_audio * level_linear
        
        # Apply dialogue level
        dialogue_level = 10 ** (self.mix_config.dialogue_level / 20)
        dialogue = dialogue * dialogue_level
        
        # Mix
        mixed = dialogue + background
        
        # Normalize and compress
        if self.mix_config.normalize:
            mixed = normalize_audio(mixed, self.mix_config.target_lufs)
        
        if self.mix_config.apply_compression:
            mixed = apply_compression(
                mixed,
                threshold_db=self.mix_config.compression_threshold,
                ratio=self.mix_config.compression_ratio,
                sample_rate=self.target_sr,
            )
        
        # Final safety clip
        mixed = np.clip(mixed, -1.0, 1.0)
        
        sf.write(str(output_path), mixed, self.target_sr)
        logger.success(f"Final mix saved: {output_path}")
        return output_path
    
    def mix_simple(
        self,
        project: ProjectData,
        original_audio: Path,
        output_path: Path,
    ) -> Path:
        """
        Simple mixing without source separation.
        Just overlays dialogue on original audio (lowered).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load original audio
        original, sr = librosa.load(str(original_audio), sr=self.target_sr)
        
        # Create dialogue track
        dialogue_track = np.zeros_like(original)
        
        for segment in project.segments:
            if segment.status != SegmentStatus.TIMING_ADJUSTED:
                continue
            
            if not segment.adjusted_audio_path or not segment.adjusted_audio_path.exists():
                continue
            
            audio, _ = librosa.load(str(segment.adjusted_audio_path), sr=self.target_sr)
            
            start_sample = int(segment.time_range.start * self.target_sr)
            end_sample = start_sample + len(audio)
            
            if end_sample > len(original):
                audio = audio[:len(original) - start_sample]
                end_sample = len(original)
            
            if start_sample < len(original):
                dialogue_track[start_sample:end_sample] = audio
        
        # Lower original audio during dialogue
        # Create mask for dialogue regions
        dialogue_mask = np.zeros_like(original)
        for segment in project.segments:
            if segment.status != SegmentStatus.TIMING_ADJUSTED:
                continue
            start = int(segment.time_range.start * self.target_sr)
            end = int(segment.time_range.end * self.target_sr)
            if start < len(dialogue_mask):
                dialogue_mask[start:min(end, len(dialogue_mask))] = 1.0
        
        # Duck original during dialogue
        duck_level = 10 ** (self.mix_config.bgm_level / 20)
        original_ducked = original * (1 - dialogue_mask * (1 - duck_level))
        
        # Mix
        mixed = original_ducked + dialogue_track
        
        # Normalize
        if self.mix_config.normalize:
            mixed = normalize_audio(mixed, self.mix_config.target_lufs)
        
        sf.write(str(output_path), mixed, self.target_sr)
        logger.success(f"Simple mix saved: {output_path}")
        return output_path


def process_mixing(project: ProjectData, config: Optional[dict] = None) -> ProjectData:
    """Process audio mixing for project."""
    cfg = config or get_config()
    
    mixer = AudioMixer(cfg)
    
    output_dir = Path(cfg.paths.temp_dir) / project.project_id / "mix"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting audio mixing")
    
    if cfg.mixing.separate_sources and project.source_audio_path:
        # Full separation workflow
        stems_dir = Path(cfg.paths.temp_dir) / project.project_id / "stems"
        stems = mixer.separate_sources(project.source_audio_path, stems_dir)
        
        # Create dialogue track
        dialogue_path = output_dir / "dialogue_track.wav"
        mixer.create_dialogue_track(project, project.total_duration, dialogue_path)
        
        # Mix
        final_audio = output_dir / "final_mix.wav"
        mixer.mix_tracks(dialogue_path, stems, final_audio)
    else:
        # Simple mixing
        final_audio = output_dir / "final_mix.wav"
        mixer.mix_simple(project, project.source_audio_path, final_audio)
    
    # Update segments with final audio path
    for segment in project.segments:
        if segment.status == SegmentStatus.TIMING_ADJUSTED:
            segment.final_audio_path = final_audio
            segment.status = SegmentStatus.MIXED
    
    logger.success("Audio mixing complete")
    return project