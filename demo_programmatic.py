#!/usr/bin/env python3
"""
Demo script showing how to use the Hindi Dubbing Pipeline programmatically.
"""
from pathlib import Path
import os
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hindi_dubbing import create_pipeline
from hindi_dubbing.config import load_config


def main():
    # Configuration
    input_video = Path("test_input.mp4")
    output_video = Path("test_output.mp4")
    project_dir = Path("project_test")
    
    # Check if HF token is set
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("⚠️  Warning: HF_TOKEN not set. Diarization will fail for gated models.")
        print("   Set HF_TOKEN environment variable for full functionality.")
    
    # Create pipeline
    print("Creating pipeline...")
    pipeline = create_pipeline()
    
    # Option 1: Run full pipeline (single command)
    print("\n=== Running Full Pipeline ===")
    try:
        pipeline.run_full_pipeline(input_video, output_video)
        print(f"✅ Done! Output: {output_video}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Option 2: Stage-by-stage (for development)
    print("\n=== Stage-by-Stage Example ===")
    print("To run stages individually:")
    print("  pipeline.create_project(input_video, project_dir)")
    print("  pipeline.run_audio_analysis()")
    print("  pipeline.run_translation()")
    print("  pipeline.run_voice_generation()")
    print("  pipeline.run_timing_adjustment()")
    print("  pipeline.run_mixing()")
    print("  pipeline.run_qc()")
    print("  pipeline.run_final_render(output_video)")
    
    # Option 3: Edit and regenerate specific segment
    print("\n=== Selective Regeneration ===")
    print("To edit a segment:")
    print("  pipeline.load_project(project_dir)")
    print("  # Edit segment in pipeline.project.segments")
    print("  pipeline.run_voice_generation()")
    print("  pipeline.run_timing_adjustment()")
    print("  pipeline.run_mixing()")
    print("  pipeline.run_qc()")


if __name__ == "__main__":
    main()