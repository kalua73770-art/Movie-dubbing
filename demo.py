#!/usr/bin/env python
"""
Demo script for Hindi Dubbing Pipeline.
Runs the pipeline on a test video to verify functionality.
"""

import sys
import subprocess
from pathlib import Path

def run_demo():
    """Run the demo pipeline."""
    workspace = Path(__file__).parent
    test_video = workspace / "test_input.mp4"
    output_video = workspace / "test_output.mp4"
    project_dir = workspace / "demo_project"
    
    if not test_video.exists():
        print(f"Test video not found: {test_video}")
        return 1
    
    print("=" * 60)
    print("Hindi Dubbing Pipeline Demo")
    print("=" * 60)
    print(f"Input:  {test_video}")
    print(f"Output: {output_video}")
    print(f"Project: {project_dir}")
    print()
    
    # Run the pipeline via CLI
    cmd = [
        sys.executable, "-m", "hindi_dubbing",
        "--verbose",
        "run",
        str(test_video),
        str(output_video),
        "--project-dir", str(project_dir),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 60)
        print("Demo completed successfully!")
        print(f"Output video: {output_video}")
        return 0
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"Demo failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print()
        print("Demo interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(run_demo())