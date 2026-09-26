"""
Audio extraction and processing utilities.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional
import numpy as np
import librosa
import soundfile as sf
from loguru import logger


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
) -> Path:
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to input video file
        output_path: Path for output audio file
        sample_rate: Target sample rate
        channels: Target number of channels
        codec: Audio codec to use
    
    Returns:
        Path to extracted audio file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", codec,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(output_path),
    ]
    
    logger.info(f"Extracting audio from {video_path} to {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    
    logger.success(f"Audio extracted: {output_path}")
    return output_path


def get_audio_info(audio_path: Path) -> dict:
    """Get audio file information."""
    info = sf.info(str(audio_path))
    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
    }


def load_audio(audio_path: Path, target_sr: Optional[int] = None) -> tuple[np.ndarray, int]:
    """
    Load audio file as numpy array.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (resample if different)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    return audio, sr


def save_audio(audio: np.ndarray, output_path: Path, sample_rate: int = 16000) -> Path:
    """Save audio array to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate)
    return output_path


def split_audio_chunks(
    audio_path: Path,
    chunk_duration: float,
    overlap: float,
    output_dir: Path,
) -> list[dict]:
    """
    Split audio into overlapping chunks for processing.
    
    Args:
        audio_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds
        output_dir: Directory to save chunks
    
    Returns:
        List of chunk info dicts with start, end, path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio, sr = load_audio(audio_path)
    total_duration = len(audio) / sr
    
    chunks = []
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    step_samples = chunk_samples - overlap_samples
    
    for i, start_sample in enumerate(range(0, len(audio), step_samples)):
        end_sample = min(start_sample + chunk_samples, len(audio))
        start_time = start_sample / sr
        end_time = end_sample / sr
        
        if end_time - start_time < 1.0:  # Skip very short chunks
            continue
        
        chunk_audio = audio[start_sample:end_sample]
        chunk_path = output_dir / f"chunk_{i:04d}.wav"
        save_audio(chunk_audio, chunk_path, sr)
        
        chunks.append({
            "index": i,
            "start": start_time,
            "end": end_time,
            "duration": end_time - start_time,
            "path": chunk_path,
            "overlap_start": overlap if i > 0 else 0,
            "overlap_end": overlap if end_sample < len(audio) else 0,
        })
        
        logger.debug(f"Created chunk {i}: {start_time:.2f}s - {end_time:.2f}s")
    
    logger.info(f"Split audio into {len(chunks)} chunks")
    return chunks


def merge_chunks_results(chunks_results: list[dict], total_duration: float) -> list[dict]:
    """
    Merge results from overlapping chunks, handling duplicates in overlap regions.
    
    Args:
        chunks_results: List of results from each chunk
        total_duration: Total audio duration
    
    Returns:
        Merged results
    """
    # This is a placeholder - actual implementation depends on result type
    # For speaker segments, we'd deduplicate in overlap regions
    # For transcripts, we'd merge text
    pass


def normalize_audio(audio: np.ndarray, target_lufs: float = -16.0) -> np.ndarray:
    """Normalize audio to target LUFS (simplified RMS normalization)."""
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        # Approximate LUFS to RMS conversion
        target_rms = 10 ** (target_lufs / 20)
        audio = audio * (target_rms / rms)
    return np.clip(audio, -1.0, 1.0)


def apply_compression(
    audio: np.ndarray,
    threshold_db: float = -18.0,
    ratio: float = 3.0,
    attack_ms: float = 5.0,
    release_ms: float = 100.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Apply dynamic range compression."""
    threshold = 10 ** (threshold_db / 20)
    
    # Simple feed-forward compressor
    envelope = np.abs(audio)
    # Smooth envelope
    attack_coeff = np.exp(-1 / (attack_ms * sample_rate / 1000))
    release_coeff = np.exp(-1 / (release_ms * sample_rate / 1000))
    
    smoothed = np.zeros_like(envelope)
    for i in range(1, len(envelope)):
        if envelope[i] > smoothed[i-1]:
            smoothed[i] = attack_coeff * smoothed[i-1] + (1 - attack_coeff) * envelope[i]
        else:
            smoothed[i] = release_coeff * smoothed[i-1] + (1 - release_coeff) * envelope[i]
    
    # Compute gain
    gain = np.ones_like(smoothed)
    over_threshold = smoothed > threshold
    gain[over_threshold] = threshold * (smoothed[over_threshold] / threshold) ** (1/ratio - 1)
    
    return audio * gain


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def trim_silence(audio: np.ndarray, top_db: float = 30, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Trim leading and trailing silence."""
    return librosa.effects.trim(audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length)[0]


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds."""
    info = sf.info(str(audio_path))
    return info.duration