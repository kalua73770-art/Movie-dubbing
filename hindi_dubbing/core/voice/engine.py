"""
Voice generation using TTS models (XTTS v2, etc.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List
import torch
import numpy as np
import soundfile as sf
from loguru import logger

from hindi_dubbing.data.models import DubSegment, SpeakerProfile, ProjectData, SegmentStatus, SpeakerGender
from hindi_dubbing.config import get_config


class VoiceEngine:
    """Voice generation engine for Hindi TTS."""
    
    def __init__(
        self,
        model_name: str = "coqui/XTTS-v2",
        language: str = "hi",
        device: Optional[str] = None,
        speaker_wav_duration: float = 6.0,
    ):
        self.model_name = model_name
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.speaker_wav_duration = speaker_wav_duration
        self._model = None
        self._speaker_embeddings: Dict[str, torch.Tensor] = {}
    
    @property
    def model(self):
        """Lazy load the TTS model."""
        if self._model is None:
            logger.info(f"Loading TTS model: {self.model_name}")
            try:
                from TTS.api import TTS
                self._model = TTS(self.model_name).to(self.device)
            except ImportError:
                # Fallback to transformers-based XTTS
                logger.warning("TTS library not available, using transformers fallback")
                self._model = self._load_transformers_xtts()
            logger.success("TTS model loaded")
        return self._model
    
    def _load_transformers_xtts(self):
        """Load XTTS using transformers directly."""
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        # This is a placeholder - actual implementation depends on model availability
        raise NotImplementedError("Transformers XTTS fallback not implemented")
    
    def extract_speaker_embedding(self, reference_audio: Path) -> torch.Tensor:
        """Extract speaker embedding from reference audio."""
        # XTTS uses speaker embeddings from reference audio
        # This is handled internally by the TTS library
        pass
    
    def generate_speech(
        self,
        text: str,
        speaker_wav: Optional[Path] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate speech from text.
        
        Args:
            text: Text to synthesize
            speaker_wav: Reference audio for voice cloning
            speaker_embedding: Pre-computed speaker embedding
            language: Target language
            **kwargs: Additional generation parameters
        
        Returns:
            Generated audio as numpy array
        """
        if not text.strip():
            return np.array([], dtype=np.float32)
        
        lang = language or self.language
        
        # Use TTS library
        if hasattr(self.model, "tts"):
            # Coqui TTS API
            if speaker_wav:
                wav = self.model.tts(
                    text=text,
                    speaker_wav=str(speaker_wav),
                    language=lang,
                    **kwargs,
                )
            elif speaker_embedding is not None:
                wav = self.model.tts(
                    text=text,
                    speaker_embedding=speaker_embedding.cpu().numpy(),
                    language=lang,
                    **kwargs,
                )
            else:
                # Use default speaker
                wav = self.model.tts(text=text, language=lang, **kwargs)
        else:
            raise NotImplementedError("TTS model interface not recognized")
        
        return np.array(wav, dtype=np.float32)
    
    def generate_for_segment(
        self,
        segment: DubSegment,
        speaker_profile: SpeakerProfile,
        output_path: Path,
        reference_audio_dir: Path,
    ) -> Path:
        """
        Generate speech for a single segment.
        
        Args:
            segment: DubSegment with hindi_text_adapted
            speaker_profile: SpeakerProfile with voice info
            output_path: Output path for generated audio
            reference_audio_dir: Directory with reference audio files
        
        Returns:
            Path to generated audio
        """
        text = segment.hindi_text_adapted or segment.hindi_text
        if not text.strip():
            logger.warning(f"Empty text for segment {segment.segment_id}")
            return output_path
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine reference audio
        ref_audio = None
        if speaker_profile.reference_audio_path and speaker_profile.reference_audio_path.exists():
            ref_audio = speaker_profile.reference_audio_path
        elif speaker_profile.voice_type == "generic":
            # Use generic voice reference
            ref_audio = reference_audio_dir / f"{speaker_profile.voice_id}.wav"
        
        try:
            audio = self.generate_speech(
                text=text,
                speaker_wav=ref_audio,
                language=self.language,
            )
            
            # Save
            sf.write(str(output_path), audio, 24000)  # XTTS outputs at 24kHz
            
            segment.generated_audio_path = output_path
            segment.status = SegmentStatus.VOICE_GENERATED
            
            logger.debug(f"Generated voice for {segment.segment_id}: {len(audio)/24000:.2f}s")
            
        except Exception as e:
            logger.error(f"Voice generation failed for {segment.segment_id}: {e}")
            segment.status = SegmentStatus.FAILED
            segment.qc_flags.append("voice_generation_failed")
        
        return output_path


class VoiceManager:
    """Manage voice assignments and reference audio."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.voice_config = self.config.voice
        self.main_characters: Dict[str, SpeakerProfile] = {}
        # Handle both dict and object access for generic_voices
        gv = self.voice_config.generic_voices
        if hasattr(gv, 'male'):
            male_voices = gv.male
            female_voices = gv.female
        else:
            male_voices = gv.get("male", ["generic_male_1", "generic_male_2", "generic_male_3"])
            female_voices = gv.get("female", ["generic_female_1", "generic_female_2", "generic_female_3"])
        self.generic_voice_pool: Dict[SpeakerGender, List[str]] = {
            SpeakerGender.MALE: male_voices,
            SpeakerGender.FEMALE: female_voices,
        }
        self.assigned_generic: Dict[str, str] = {}  # speaker_id -> voice_id
    
    def analyze_speakers(self, project: ProjectData) -> ProjectData:
        """
        Analyze speaker frequencies and assign voice types.
        Main characters get dedicated voices, others get generic pool.
        """
        project.update_speaker_stats()
        
        # Sort speakers by frequency
        sorted_speakers = sorted(
            project.speakers.items(),
            key=lambda x: x[1].frequency_percent,
            reverse=True
        )
        
        main_threshold = self.voice_config.main_character_threshold * 100  # Convert to percent
        max_main = self.voice_config.max_main_characters
        
        main_count = 0
        
        for speaker_id, profile in sorted_speakers:
            is_main = (profile.frequency_percent >= main_threshold and main_count < max_main)
            
            if is_main:
                profile.voice_type = "main"
                profile.voice_id = f"main_{speaker_id}"
                main_count += 1
                logger.info(f"Main character: {speaker_id} ({profile.frequency_percent:.1f}%) -> {profile.voice_id}")
            else:
                profile.voice_type = "generic"
                # Assign from generic pool based on gender
                gender = profile.gender
                if gender == SpeakerGender.UNKNOWN:
                    gender = SpeakerGender.MALE  # Default
                
                pool = self.generic_voice_pool[gender]
                # Round-robin assignment
                pool_idx = len([p for p in project.speakers.values() 
                               if p.voice_type == "generic" and p.gender == gender]) % len(pool)
                profile.voice_id = pool[pool_idx]
                logger.info(f"Generic character: {speaker_id} ({profile.frequency_percent:.1f}%) -> {profile.voice_id}")
        
        return project
    
    def extract_reference_audio(
        self,
        project: ProjectData,
        source_audio: Path,
        output_dir: Path,
    ) -> ProjectData:
        """
        Extract reference audio for each main character.
        Uses the longest clean segment for each speaker.
        """
        import librosa
        import soundfile as sf
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio, sr = librosa.load(str(source_audio), sr=24000)
        
        for speaker_id, profile in project.speakers.items():
            if profile.voice_type != "main":
                continue
            
            # Get all segments for this speaker
            segments = project.get_segments_by_speaker(speaker_id)
            if not segments:
                continue
            
            # Find longest segment (or combine top N)
            segments.sort(key=lambda s: s.time_range.duration, reverse=True)
            
            # Use top 3 segments up to target duration
            target_samples = int(self.voice_config.speaker_wav_duration * sr)
            combined = np.array([], dtype=np.float32)
            
            for seg in segments:
                if len(combined) >= target_samples:
                    break
                start = int(seg.time_range.start * sr)
                end = int(seg.time_range.end * sr)
                combined = np.concatenate([combined, audio[start:end]])
            
            # Trim to target duration
            combined = combined[:target_samples]
            
            # Save reference
            ref_path = output_dir / f"{profile.voice_id}.wav"
            sf.write(str(ref_path), combined, sr)
            
            profile.reference_audio_path = ref_path
            logger.info(f"Extracted reference for {speaker_id}: {len(combined)/sr:.2f}s")
        
        return project


def process_voice_generation(project: ProjectData, config: Optional[dict] = None) -> ProjectData:
    """
    Process voice generation for all translated segments.
    """
    cfg = config or get_config()
    
    engine = VoiceEngine(
        model_name=cfg.voice.tts_model,
        language=cfg.voice.tts_language,
        speaker_wav_duration=cfg.voice.speaker_wav_duration,
    )
    
    manager = VoiceManager(cfg)
    
    # Analyze and assign voices
    project = manager.analyze_speakers(project)
    
    # Extract reference audio for main characters
    ref_dir = Path(cfg.paths.cache_dir) / "references" / project.project_id
    project = manager.extract_reference_audio(
        project,
        project.source_audio_path,
        ref_dir,
    )
    
    # Generate voices for all segments
    output_dir = Path(cfg.paths.temp_dir) / project.project_id / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating voices for {len(project.segments)} segments")
    
    for segment in project.segments:
        if segment.status != SegmentStatus.TRANSLATED:
            continue
        
        speaker_profile = project.speakers.get(segment.speaker_id)
        if not speaker_profile:
            logger.warning(f"No speaker profile for {segment.speaker_id}")
            segment.status = SegmentStatus.FAILED
            continue
        
        output_path = output_dir / f"{segment.segment_id}.wav"
        
        engine.generate_for_segment(
            segment,
            speaker_profile,
            output_path,
            ref_dir,
        )
    
    logger.success("Voice generation complete")
    return project