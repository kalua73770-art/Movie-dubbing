"""
Final video rendering with Hindi audio.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
from loguru import logger

from hindi_dubbing.data.models import ProjectData
from hindi_dubbing.config import get_config


def render_final_video(
    project: ProjectData,
    output_path: Path,
    config: Optional[dict] = None,
) -> Path:
    """
    Render final video with Hindi audio track.
    
    Uses ffmpeg to combine original video with new audio.
    """
    cfg = config or get_config()
    render_config = cfg.render
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find final mixed audio
    mixed_audio = None
    for segment in project.segments:
        if segment.final_audio_path and segment.final_audio_path.exists():
            mixed_audio = segment.final_audio_path
            break
    
    if not mixed_audio:
        # Try to find in temp dir
        temp_mix = Path(cfg.paths.temp_dir) / project.project_id / "mix" / "final_mix.wav"
        if temp_mix.exists():
            mixed_audio = temp_mix
    
    if not mixed_audio:
        raise FileNotFoundError("No final mixed audio found")
    
    logger.info(f"Rendering final video: {output_path}")
    logger.info(f"  Video: {project.source_video_path}")
    logger.info(f"  Audio: {mixed_audio}")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(project.source_video_path),
        "-i", str(mixed_audio),
        "-c:v", render_config.video_codec,
        "-c:a", render_config.audio_codec,
        "-b:a", render_config.audio_bitrate,
        "-preset", render_config.preset,
        "-crf", str(render_config.crf),
        "-map", "0:v:0",  # Video from first input
        "-map", "1:a:0",  # Audio from second input
        "-shortest",      # End when shortest stream ends
        str(output_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    
    logger.success(f"Final video rendered: {output_path}")
    return output_path


def render_preview(
    project: ProjectData,
    output_path: Path,
    start_time: float = 0,
    duration: float = 60,
    config: Optional[dict] = None,
) -> Path:
    """Render a preview clip for quick review."""
    cfg = config or get_config()
    render_config = cfg.render
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mixed_audio = None
    for segment in project.segments:
        if segment.final_audio_path and segment.final_audio_path.exists():
            mixed_audio = segment.final_audio_path
            break
    
    if not mixed_audio:
        temp_mix = Path(cfg.paths.temp_dir) / project.project_id / "mix" / "final_mix.wav"
        if temp_mix.exists():
            mixed_audio = temp_mix
    
    if not mixed_audio:
        raise FileNotFoundError("No final mixed audio found")
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", str(project.source_video_path),
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", str(mixed_audio),
        "-c:v", render_config.video_codec,
        "-c:a", render_config.audio_codec,
        "-b:a", render_config.audio_bitrate,
        "-preset", "fast",
        "-crf", "23",
        "-map", "0:v:0",
        "-map", "1:a:0",
        str(output_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    
    logger.success(f"Preview rendered: {output_path}")
    return output_path