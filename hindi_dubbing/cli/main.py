"""
Command-line interface for Hindi Dubbing Pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from loguru import logger

from hindi_dubbing.pipeline.orchestrator import create_pipeline, Pipeline
from hindi_dubbing.config import load_config
from hindi_dubbing.data.models import ProjectData, SegmentStatus, QCFlag


console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logger.remove()
    logger.add(
        lambda msg: console.print(msg, end=""),
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool) -> None:
    """Hindi Dubbing Pipeline - AI-assisted movie/anime dubbing."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = Path(config) if config else None
    ctx.obj["verbose"] = verbose
    setup_logging("DEBUG" if verbose else "INFO")


@cli.command()
@click.argument("input_video", type=click.Path(exists=True))
@click.argument("output_video", type=click.Path())
@click.option("--project-dir", "-p", type=click.Path(), help="Project directory")
@click.pass_context
def run(ctx: click.Context, input_video: str, output_video: str, project_dir: Optional[str]) -> None:
    """Run full dubbing pipeline on a video."""
    config_path = ctx.obj["config_path"]
    pipeline = create_pipeline(config_path)
    
    input_path = Path(input_video)
    output_path = Path(output_video)
    
    if project_dir:
        proj_dir = Path(project_dir)
    else:
        proj_dir = output_path.parent / f"project_{output_path.stem}"
    
    console.print(f"[bold]Starting full pipeline[/bold]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Project: {proj_dir}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running pipeline...", total=8)
        
        def callback(stage: str, message: str, prog: float):
            progress.update(task, description=message, advance=prog)
        
        pipeline.add_callback(callback)
        pipeline.run_full_pipeline(input_path, output_path)
    
    console.print("[bold green]Done![/bold green] Output:", str(output_path))


@cli.command()
@click.argument("input_video", type=click.Path(exists=True))
@click.argument("project_dir", type=click.Path())
@click.pass_context
def analyze(ctx: click.Context, input_video: str, project_dir: str) -> None:
    """Run audio analysis (diarization + transcription) only."""
    config_path = ctx.obj["config_path"]
    pipeline = create_pipeline(config_path)
    
    pipeline.create_project(Path(input_video), Path(project_dir))
    pipeline.run_audio_analysis()
    
    console.print(f"[bold green]Analysis complete![/bold] Project saved to {project_dir}")


@cli.command()
@click.argument("project_dir", type=click.Path(exists=True))
@click.pass_context
def translate(ctx: click.Context, project_dir: str) -> None:
    """Run Hindi translation on analyzed project."""
    config_path = ctx.obj["config_path"]
    pipeline = create_pipeline(config_path)
    
    pipeline.load_project(Path(project_dir))
    pipeline.run_translation()
    
    console.print("[bold green]Translation complete![/bold]")


@cli.command()
@click.argument("project_dir", type=click.Path(exists=True))
@click.pass_context
def generate(ctx: click.Context, project_dir: str) -> None:
    """Generate Hindi voices for translated segments."""
    config_path = ctx.obj["config_path"]
    pipeline = create_pipeline(config_path)
    
    pipeline.load_project(Path(project_dir))
    pipeline.run_voice_generation()
    
    console.print("[bold green]Voice generation complete![/bold]")


@cli.command()
@click.argument("project_dir", type=click.Path(exists=True))
@click.pass_context
def adjust(ctx: click.Context, project_dir: str) -> None:
    """Adjust timing of generated voices."""
    config_path = ctx.obj["config_path"]
    pipeline = create_pipeline(config_path)
    
    pipeline.load_project(Path(project_dir))
    pipeline.run_timing_adjustment()
    
    console.print("[bold green]Timing adjustment complete![/bold]")


@cli.command()
@click.argument("project_dir", type=click.Path(exists=True))
@click.pass_context
def mix(ctx: click.Context, project_dir: str) -> None:
    """Mix Hindi dialogue with background audio."""
    config_path = ctx.obj["config_path"]
    pipeline = create_pipeline(config_path)
    
    pipeline.load_project(Path(project_dir))
    pipeline.run_mixing()
    
    console.print("[bold green]Mixing complete![/bold]")


@cli.command()
@click.argument("project_dir", type=click.Path(exists=True))
@click.pass_context
def qc(ctx: click.Context, project_dir: str) -> None:
    """Run quality control checks."""
    config_path = ctx.obj["config_path"]
    pipeline = create_pipeline(config_path)
    
    pipeline.load_project(Path(project_dir))
    pipeline.run_qc()
    
    # Show summary
    project = pipeline.project
    summary = project.qc_summary
    
    table = Table(title="QC Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Segments", str(summary.get("total_segments", 0)))
    table.add_row("Clean Segments", str(summary.get("clean_segments", 0)))
    table.add_row("Flagged Segments", str(summary.get("flagged_segments", 0)))
    
    console.print(table)
    
    if summary.get("flag_counts"):
        flag_table = Table(title="Flag Counts")
        flag_table.add_column("Flag", style="cyan")
        flag_table.add_column("Count", style="magenta")
        for flag, count in summary["flag_counts"].items():
            flag_table.add_row(flag, str(count))
        console.print(flag_table)


@cli.command()
@click.argument("project_dir", type=click.Path(exists=True))
@click.argument("output_video", type=click.Path())
@click.pass_context
def render(ctx: click.Context, project_dir: str, output_video: str) -> None:
    """Render final video with Hindi audio."""
    config_path = ctx.obj["config_path"]
    pipeline = create_pipeline(config_path)
    
    pipeline.load_project(Path(project_dir))
    pipeline.run_final_render(Path(output_video))
    
    console.print(f"[bold green]Final video rendered![/bold] {output_video}")


@cli.command()
@click.argument("project_dir", type=click.Path(exists=True))
@click.argument("output_video", type=click.Path())
@click.option("--start", "-s", type=float, default=0, help="Start time in seconds")
@click.option("--duration", "-d", type=float, default=60, help="Duration in seconds")
@click.pass_context
def preview(ctx: click.Context, project_dir: str, output_video: str, start: float, duration: float) -> None:
    """Render a preview clip."""
    config_path = ctx.obj["config_path"]
    pipeline = create_pipeline(config_path)
    
    pipeline.load_project(Path(project_dir))
    pipeline.render_preview(Path(output_video), start, duration)
    
    console.print(f"[bold green]Preview rendered![/bold] {output_video}")


@cli.command()
@click.argument("project_dir", type=click.Path(exists=True))
@click.option("--status", type=click.Choice([s.value for s in SegmentStatus]), help="Filter by status")
@click.option("--speaker", help="Filter by speaker ID")
@click.pass_context
def list_segments(ctx: click.Context, project_dir: str, status: Optional[str], speaker: Optional[str]) -> None:
    """List segments in project."""
    pipeline = create_pipeline(ctx.obj["config_path"])
    pipeline.load_project(Path(project_dir))
    
    segments = pipeline.project.segments
    
    if status:
        segments = [s for s in segments if s.status.value == status]
    if speaker:
        segments = [s for s in segments if s.speaker_id == speaker]
    
    table = Table(title=f"Segments ({len(segments)} total)")
    table.add_column("ID", style="cyan")
    table.add_column("Speaker", style="magenta")
    table.add_column("Character", style="green")
    table.add_column("Time", style="yellow")
    table.add_column("Status", style="blue")
    table.add_column("Original Text", style="white")
    table.add_column("Hindi Text", style="white")
    
    for seg in segments[:50]:  # Limit display
        table.add_row(
            seg.segment_id,
            seg.speaker_id,
            seg.character_name or "-",
            f"{seg.time_range.start:.1f}-{seg.time_range.end:.1f}s",
            seg.status.value,
            seg.original_text[:40] + "..." if len(seg.original_text) > 40 else seg.original_text,
            seg.hindi_text_adapted[:40] + "..." if len(seg.hindi_text_adapted) > 40 else seg.hindi_text_adapted,
        )
    
    console.print(table)
    
    if len(segments) > 50:
        console.print(f"[dim]... and {len(segments) - 50} more segments[/dim]")


@cli.command()
@click.argument("project_dir", type=click.Path(exists=True))
@click.argument("segment_id", type=str)
@click.option("--hindi-text", help="New Hindi text")
@click.pass_context
def edit_segment(ctx: click.Context, project_dir: str, segment_id: str, hindi_text: Optional[str]) -> None:
    """Edit a segment's Hindi text and regenerate."""
    pipeline = create_pipeline(ctx.obj["config_path"])
    pipeline.load_project(Path(project_dir))
    
    segment = next((s for s in pipeline.project.segments if s.segment_id == segment_id), None)
    if not segment:
        console.print(f"[red]Segment not found: {segment_id}[/red]")
        return
    
    if hindi_text:
        segment.hindi_text = hindi_text
        segment.hindi_text_adapted = ""  # Will be re-matched
        segment.status = SegmentStatus.TRANSLATED
        console.print(f"Updated Hindi text for {segment_id}")
    
    # Re-run from translation for this segment
    from hindi_dubbing.core.translation.engine import DurationMatcher
    from hindi_dubbing.config import get_config
    
    config = get_config()
    matcher = DurationMatcher(config)
    segment.hindi_text_adapted = matcher.match_duration(segment, segment.time_range.duration)
    
    pipeline.run_voice_generation()
    pipeline.run_timing_adjustment()
    pipeline.run_mixing()
    pipeline.run_qc()
    
    console.print(f"[bold green]Segment {segment_id} regenerated![/bold]")


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show pipeline information."""
    config = load_config(ctx.obj["config_path"])
    
    console.print("[bold]Hindi Dubbing Pipeline[/bold]")
    console.print(f"Version: {config.project.version}")
    console.print(f"Chunk Duration: {config.project.chunk_duration_seconds}s")
    console.print(f"Chunk Overlap: {config.project.chunk_overlap_seconds}s")
    console.print(f"Diarization Model: {config.diarization.model}")
    console.print(f"Transcription Model: {config.transcription.model}")
    console.print(f"Translation Model: {config.translation.model}")
    console.print(f"TTS Model: {config.voice.tts_model}")
    console.print(f"Source Separation: {config.mixing.separate_sources}")


if __name__ == "__main__":
    cli()