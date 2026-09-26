"""
Speech-to-text transcription using Whisper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import whisper
import torch
from loguru import logger

from hindi_dubbing.data.models import SpeakerSegment, DubSegment, TimeRange


class TranscriptionEngine:
    """Transcription engine using OpenAI Whisper."""
    
    def __init__(
        self,
        model_name: str = "large-v3",
        language: str = "en",
        device: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kwargs = kwargs
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self._model = whisper.load_model(self.model_name, device=self.device)
            logger.success("Whisper model loaded")
        return self._model
    
    def transcribe(
        self,
        audio_path: Path,
        segments: list[SpeakerSegment],
    ) -> list[DubSegment]:
        """
        Transcribe audio segments.
        
        Args:
            audio_path: Path to full audio file
            segments: List of speaker segments from diarization
        
        Returns:
            List of DubSegments with transcriptions
        """
        logger.info(f"Transcribing {len(segments)} segments")
        
        # Load full audio
        import librosa
        audio, sr = librosa.load(str(audio_path), sr=16000)
        
        dub_segments = []
        
        for i, seg in enumerate(segments):
            # Extract segment audio
            start_sample = int(seg.time_range.start * sr)
            end_sample = int(seg.time_range.end * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Save temporary segment
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, segment_audio, sr)
                tmp_path = Path(tmp.name)
            
            try:
                # Transcribe
                result = self.model.transcribe(
                    str(tmp_path),
                    language=self.language,
                    **self.kwargs,
                )
                
                text = result["text"].strip()
                
                # Create dub segment
                dub_segment = DubSegment(
                    segment_id=f"seg_{i:06d}",
                    speaker_id=seg.speaker_id,
                    time_range=seg.time_range,
                    original_text=text,
                    original_language=self.language,
                    confidence=seg.confidence,
                )
                
                dub_segments.append(dub_segment)
                logger.debug(f"Segment {i}: {text[:50]}...")
                
            except Exception as e:
                logger.error(f"Transcription failed for segment {i}: {e}")
                # Create empty segment
                dub_segment = DubSegment(
                    segment_id=f"seg_{i:06d}",
                    speaker_id=seg.speaker_id,
                    time_range=seg.time_range,
                    original_text="",
                    original_language=self.language,
                    confidence=0.0,
                    status=DubSegment.__dataclass_fields__["status"].default,
                )
                dub_segments.append(dub_segment)
            finally:
                tmp_path.unlink(missing_ok=True)
        
        logger.success(f"Transcription complete: {len(dub_segments)} segments")
        return dub_segments
    
    def transcribe_full_audio(self, audio_path: Path) -> dict:
        """
        Transcribe full audio at once (for comparison/validation).
        
        Returns:
            Whisper result dict with segments
        """
        logger.info(f"Transcribing full audio: {audio_path}")
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            **self.kwargs,
        )
        return result


def align_transcription_to_segments(
    whisper_result: dict,
    diarization_segments: list[SpeakerSegment],
) -> list[DubSegment]:
    """
    Align Whisper word-level timestamps to diarization segments.
    This provides more accurate text-to-speaker alignment.
    """
    dub_segments = []
    
    whisper_segments = whisper_result.get("segments", [])
    whisper_words = []
    for ws in whisper_segments:
        for word in ws.get("words", []):
            whisper_words.append({
                "word": word["word"],
                "start": word["start"],
                "end": word["end"],
            })
    
    for i, diar_seg in enumerate(diarization_segments):
        # Find words that overlap with this diarization segment
        seg_words = []
        for word in whisper_words:
            if (word["start"] < diar_seg.time_range.end and
                word["end"] > diar_seg.time_range.start):
                seg_words.append(word["word"])
        
        text = " ".join(seg_words).strip()
        
        dub_segment = DubSegment(
            segment_id=f"seg_{i:06d}",
            speaker_id=diar_seg.speaker_id,
            time_range=diar_seg.time_range,
            original_text=text,
            original_language="en",
            confidence=diar_seg.confidence,
        )
        
        dub_segments.append(dub_segment)
    
    return dub_segments