#!/usr/bin/env python3
"""
Simple test to verify imports work.
"""
import sys
sys.path.insert(0, '.')

# Test imports
from hindi_dubbing import (
    Config, load_config, get_config,
    ProjectData, DubSegment, SpeakerProfile,
    Pipeline, create_pipeline,
)

print("All imports successful!")

# Test config
config = get_config()
print(f"Config loaded: {config.project.name} v{config.project.version}")

# Test pipeline creation
pipeline = create_pipeline()
print("Pipeline created successfully!")

print("Basic tests passed!")