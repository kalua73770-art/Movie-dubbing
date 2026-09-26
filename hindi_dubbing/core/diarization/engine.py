"""
Speaker diarization using pyannote.audio.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from loguru import logger

from hindi_dubbing.data.models import SpeakerSegment, TimeRange


class DiarizationEngine:
    """Speaker diarization engine using pyannote.audio."""
    
    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        min_speakers: int = 1,
        max_speakers: int = 20,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token
        self._pipeline: Optional[Pipeline] = None
    
    @property
    def pipeline(self) -> Pipeline:
        """Lazy load the pipeline."""
        if self._pipeline is None:
            logger.info(f"Loading diarization model: {self.model_name}")
            try:
                self._pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    token=self.hf_token,
                )
                self._pipeline.to(torch.device(self.device))
                logger.success("Diarization model loaded")
            except Exception as e:
                logger.error(f"Failed to load diarization model: {e}")
                logger.warning("Diarization will not work without valid HF token for gated models")
                raise
        return self._pipeline
    
    def diarize(self, audio_path: Path) -> Annotation:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            pyannote Annotation with speaker segments
        """
        logger.info(f"Running diarization on {audio_path}")
        
        # Run diarization
        diarization = self.pipeline(
            str(audio_path),
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )
        
        logger.info(f"Found {len(diarization.labels())} speakers")
        for label in diarization.labels():
            duration = sum(seg.duration for seg in diarization.label_timeline(label))
            logger.debug(f"  {label}: {duration:.2f}s")
        
        return diarization
    
    def diarize_chunks(
        self,
        chunk_paths: list[Path],
        chunk_infos: list[dict],
    ) -> list[SpeakerSegment]:
        """
        Diarize multiple chunks and merge results.
        
        Args:
            chunk_paths: List of chunk audio file paths
            chunk_infos: List of chunk info dicts with start, end, overlap info
        
        Returns:
            Merged list of SpeakerSegments
        """
        all_segments = []
        
        for i, (chunk_path, chunk_info) in enumerate(zip(chunk_paths, chunk_infos)):
            logger.info(f"Diarizing chunk {i+1}/{len(chunk_paths)}")
            
            try:
                annotation = self.diarize(chunk_path)
                
                # Convert to SpeakerSegments with global timestamps
                for segment, _, label in annotation.itertracks(yield_label=True):
                    global_start = chunk_info["start"] + segment.start
                    global_end = chunk_info["start"] + segment.end
                    
                    # Skip segments in overlap region (except first chunk)
                    if i > 0 and segment.start < chunk_info["overlap_start"]:
                        continue
                    # Skip segments in overlap region at end (except last chunk)
                    if i < len(chunk_paths) - 1 and segment.end > chunk_info["duration"] - chunk_info["overlap_end"]:
                        continue
                    
                    all_segments.append(SpeakerSegment(
                        speaker_id=label,
                        time_range=TimeRange(global_start, global_end),
                        confidence=1.0,  # pyannote doesn't provide per-segment confidence
                    ))
                    
            except Exception as e:
                logger.error(f"Diarization failed for chunk {i}: {e}")
                continue
        
        # Sort by start time
        all_segments.sort(key=lambda s: s.time_range.start)
        
        # Merge adjacent segments from same speaker
        merged = self._merge_adjacent_segments(all_segments)
        
        logger.success(f"Diarization complete: {len(merged)} segments, {len(set(s.speaker_id for s in merged))} speakers")
        return merged
    
    def _merge_adjacent_segments(
        self,
        segments: list[SpeakerSegment],
        max_gap: float = 0.5,
    ) -> list[SpeakerSegment]:
        """Merge adjacent segments from the same speaker."""
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for segment in segments[1:]:
            last = merged[-1]
            
            # If same speaker and close together, merge
            if (segment.speaker_id == last.speaker_id and
                segment.time_range.start - last.time_range.end <= max_gap):
                last.time_range = TimeRange(last.time_range.start, segment.time_range.end)
            else:
                merged.append(segment)
        
        return merged


def classify_speaker_gender(
    audio_path: Path,
    segments: list[SpeakerSegment],
    sample_rate: int = 16000,
) -> dict[str, str]:
    """
    Classify speaker gender based on pitch characteristics.
    Simple heuristic using fundamental frequency.
    """
    import librosa
    import numpy as np
    
    audio, sr = librosa.load(str(audio_path), sr=sample_rate)
    
    gender_map = {}
    
    for speaker_id in set(s.speaker_id for s in segments):
        # Collect all audio for this speaker
        speaker_audio = []
        for seg in segments:
            if seg.speaker_id == speaker_id:
                start_sample = int(seg.time_range.start * sr)
                end_sample = int(seg.time_range.end * sr)
                speaker_audio.append(audio[start_sample:end_sample])
        
        if not speaker_audio:
            gender_map[speaker_id] = "unknown"
            continue
        
        speaker_audio = np.concatenate(speaker_audio)
        
        # Estimate fundamental frequency using pyin
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                speaker_audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
            )
            
            # Median F0 of voiced frames
            voiced_f0 = f0[voiced_flag]
            if len(voiced_f0) > 0:
                median_f0 = np.median(voiced_f0)
                
                # Heuristic thresholds
                if median_f0 < 165:
                    gender = "male"
                elif median_f0 > 195:
                    gender = "female"
                else:
                    gender = "unknown"
            else:
                gender = "unknown"
        except Exception:
            gender = "unknown"
        
        gender_map[speaker_id] = gender
        logger.debug(f"Speaker {speaker_id}: median F0 = {median_f0:.1f} Hz -> {gender}")
    
    return gender_map