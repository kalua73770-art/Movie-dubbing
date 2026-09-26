#!/usr/bin/env python3
"""
Test script demonstrating the pipeline data flow with mock data.
This runs without requiring external models or HF token.
"""
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from hindi_dubbing.data.models import (
    ProjectData, DubSegment, SpeakerProfile, 
    TimeRange, SpeakerGender, SegmentStatus, QCFlag
)
from hindi_dubbing.core.translation.engine import DurationMatcher
from hindi_dubbing.config import get_config


def create_mock_project() -> ProjectData:
    """Create a mock project with sample data."""
    project = ProjectData(
        project_id="demo_123",
        source_video_path=Path("movie.mp4"),
        source_audio_path=Path("movie_audio.wav"),
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        total_duration=300.0,
        sample_rate=16000,
        channels=1,
        current_stage="initialized",
    )
    
    # Add speakers
    project.speakers["SPEAKER_01"] = SpeakerProfile(
        speaker_id="SPEAKER_01",
        character_name="Hero",
        gender=SpeakerGender.MALE,
        voice_id="main_SPEAKER_01",
        voice_type="main",
    )
    project.speakers["SPEAKER_02"] = SpeakerProfile(
        speaker_id="SPEAKER_02",
        character_name="Villain",
        gender=SpeakerGender.MALE,
        voice_id="main_SPEAKER_02",
        voice_type="main",
    )
    project.speakers["SPEAKER_03"] = SpeakerProfile(
        speaker_id="SPEAKER_03",
        character_name="Sidekick",
        gender=SpeakerGender.FEMALE,
        voice_id="generic_female_1",
        voice_type="generic",
    )
    
    # Add segments
    segments = [
        DubSegment(
            segment_id="seg_000001",
            speaker_id="SPEAKER_01",
            character_name="Hero",
            time_range=TimeRange(10.0, 14.5),
            original_text="What are you doing here?",
            original_language="en",
            status=SegmentStatus.TRANSCRIBED,
        ),
        DubSegment(
            segment_id="seg_000002",
            speaker_id="SPEAKER_02",
            character_name="Villain",
            time_range=TimeRange(15.0, 20.0),
            original_text="I'm here to destroy everything you love.",
            original_language="en",
            status=SegmentStatus.TRANSCRIBED,
        ),
        DubSegment(
            segment_id="seg_000003",
            speaker_id="SPEAKER_01",
            character_name="Hero",
            time_range=TimeRange(21.0, 25.5),
            original_text="Not on my watch!",
            original_language="en",
            status=SegmentStatus.TRANSCRIBED,
        ),
        DubSegment(
            segment_id="seg_000004",
            speaker_id="SPEAKER_03",
            character_name="Sidekick",
            time_range=TimeRange(26.0, 30.0),
            original_text="Hero, behind you!",
            original_language="en",
            status=SegmentStatus.TRANSCRIBED,
        ),
    ]
    project.segments = segments
    project.update_speaker_stats()
    
    return project


def test_duration_matching():
    """Test the duration matching logic."""
    print("=== Testing Duration Matching ===\n")
    
    config = get_config()
    matcher = DurationMatcher(config)
    
    project = create_mock_project()
    
    # Mock Hindi translations (what NLLB would produce)
    hindi_translations = {
        "seg_000001": "तुम यहाँ क्या कर रहे हो?",
        "seg_000002": "मैं यहाँ सब कुछ नष्ट करने आया हूँ जो तुम प्यार करते हो।",
        "seg_000003": "मेरी नजर में नहीं!",
        "seg_000004": "हीरो, पीछे देखो!",
    }
    
    for segment in project.segments:
        segment.hindi_text = hindi_translations[segment.segment_id]
        adapted = matcher.match_duration(segment, segment.time_range.duration)
        segment.hindi_text_adapted = adapted
        
        print(f"Segment: {segment.segment_id}")
        print(f"  Speaker: {segment.character_name}")
        print(f"  Original: {segment.original_text}")
        print(f"  Hindi:    {segment.hindi_text}")
        print(f"  Adapted:  {segment.hindi_text_adapted}")
        print(f"  Duration: {segment.time_range.duration:.1f}s, Ratio: {segment.duration_ratio:.2f}")
        print()


def test_qc_flags():
    """Test QC flagging."""
    print("=== Testing QC Flags ===\n")
    
    from hindi_dubbing.core.qc.engine import QualityController
    from hindi_dubbing.config import get_config
    
    config = get_config()
    qc = QualityController(config)
    
    project = create_mock_project()
    
    # Add some problematic segments
    project.segments[0].hindi_text = "तुम यहाँ क्या कर रहे हो यह बताओ मुझे अभी"
    project.segments[0].hindi_text_adapted = project.segments[0].hindi_text
    project.segments[0].duration_ratio = 2.5  # Too long
    
    project.segments[1].confidence = 0.5  # Low confidence
    
    project.segments[2].time_range = TimeRange(100.0, 100.05)  # Too short
    
    project = qc.run_qc(project)
    
    print(f"Total segments: {project.qc_summary['total_segments']}")
    print(f"Flagged segments: {project.qc_summary['flagged_segments']}")
    print(f"Clean segments: {project.qc_summary['clean_segments']}")
    print("\nFlag counts:")
    for flag, count in project.qc_summary['flag_counts'].items():
        print(f"  {flag}: {count}")
    
    print("\nFlagged segments detail:")
    for segment in project.segments:
        if segment.qc_flags:
            print(f"  {segment.segment_id}: {', '.join(f.value for f in segment.qc_flags)}")


def test_json_serialization():
    """Test JSON save/load."""
    print("=== Testing JSON Serialization ===\n")
    
    project = create_mock_project()
    
    # Add some Hindi text
    project.segments[0].hindi_text = "तुम यहाँ क्या कर रहे हो?"
    project.segments[0].hindi_text_adapted = "तुम यहाँ क्या कर रहे हो?"
    project.segments[0].status = SegmentStatus.TRANSLATED
    
    # Save to JSON
    output_path = Path("demo_project.json")
    project.save(output_path)
    print(f"Saved project to {output_path}")
    
    # Load from JSON
    loaded = ProjectData.load(output_path)
    print(f"Loaded project: {loaded.project_id}")
    print(f"Speakers: {list(loaded.speakers.keys())}")
    print(f"Segments: {len(loaded.segments)}")
    
    # Verify data integrity
    assert loaded.project_id == project.project_id
    assert len(loaded.segments) == len(project.segments)
    assert loaded.segments[0].hindi_text == project.segments[0].hindi_text
    print("✅ JSON serialization works correctly")


def main():
    print("Hindi Dubbing Pipeline - Data Flow Demo\n")
    print("=" * 50 + "\n")
    
    test_duration_matching()
    test_qc_flags()
    test_json_serialization()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✅")
    print("\nTo run the full pipeline with real models:")
    print("  1. Set HF_TOKEN environment variable")
    print("  2. Run: python3 -m hindi_dubbing run --input movie.mp4 --output dubbed.mp4")


if __name__ == "__main__":
    main()