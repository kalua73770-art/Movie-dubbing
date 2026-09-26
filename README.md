# Hindi Dubbing Pipeline

AI-assisted Hindi movie/anime dubbing pipeline with audio-first approach.

## Features

- **Audio-First Processing**: No frame-by-frame video analysis needed
- **Speaker Diarization**: Automatic speaker detection using pyannote.audio
- **Character Mapping**: Persistent speaker-to-character mapping
- **Hindi Translation**: NLLB-200 for natural Hindi adaptation with duration matching
- **Voice Generation**: XTTS-v2 for main characters, generic pool for minor characters
- **Timing Adjustment**: Automatic time-stretching to match original dialogue timing
- **Source Separation**: Demucs for BGM/SFX preservation
- **Quality Control**: Automated QC with flagging system
- **Selective Regeneration**: Edit and regenerate individual segments

## Pipeline Stages

```
Original Movie
      ↓
Audio Extraction (ffmpeg)
      ↓
Speech Detection + Speaker Diarization (pyannote)
      ↓
Speaker → Character Mapping (persistent JSON)
      ↓
Transcription (Whisper)
      ↓
Hindi Translation + Duration Matching (NLLB-200)
      ↓
Voice Generation (XTTS-v2 / generic pool)
      ↓
Timing Adjustment (librosa time-stretch)
      ↓
Source Separation + Mixing (Demucs)
      ↓
Quality Control (automated flags)
      ↓
Final Render (ffmpeg)
```

## Installation

```bash
# Install system dependencies
sudo apt-get update && sudo apt-get install -y ffmpeg

# Install Python package
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Configuration

### Hugging Face Token (Required for Diarization)

The speaker diarization model (pyannote/speaker-diarization-3.1) is gated on Hugging Face.

1. Get a token from [Hugging Face](https://huggingface.co/settings/tokens)
2. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Set the token:

```bash
export HF_TOKEN=your_token_here
# or
export HUGGINGFACE_TOKEN=your_token_here
```

### Configuration File

Copy `config/default.yaml` to `config/local.yaml` and customize:

```yaml
project:
  chunk_duration_seconds: 600  # 10 minutes
  chunk_overlap_seconds: 30

diarization:
  model: "pyannote/speaker-diarization-3.1"
  min_speakers: 1
  max_speakers: 20

transcription:
  model: "large-v3"

translation:
  model: "facebook/nllb-200-distilled-600M"

voice:
  tts_model: "coqui/XTTS-v2"
  main_character_threshold: 0.05  # 5% of speech time
  max_main_characters: 20

mixing:
  separate_sources: true
  separation_model: "htdemucs"
```

## Usage

### Full Pipeline (Recommended)

```bash
# Run complete dubbing pipeline
hindi-dubbing run --input movie.mp4 --output dubbed_movie.mp4
```

### Stage-by-Stage (For Development/Testing)

```bash
# 1. Create project and run audio analysis
hindi-dubbing analyze --input movie.mp4 --project-dir project/

# 2. Translate to Hindi
hindi-dubbing translate --project-dir project/

# 3. Generate Hindi voices
hindi-dubbing generate --project-dir project/

# 4. Adjust timing
hindi-dubbing adjust --project-dir project/

# 5. Mix with background audio
hindi-dubbing mix --project-dir project/

# 6. Run quality control
hindi-dubbing qc --project-dir project/

# 7. Render final video
hindi-dubbing render --project-dir project/ --output dubbed_movie.mp4
```

### Utility Commands

```bash
# List segments in project
hindi-dubbing list-segments --project-dir project/

# Render preview clip (60 seconds from start)
hindi-dubbing preview --project-dir project/ --output preview.mp4

# Edit a segment's Hindi text and regenerate
hindi-dubbing edit-segment --project-dir project/ --segment-id seg_000001 --hindi-text "नया हिंदी डायलॉग"

# Show pipeline info
hindi-dubbing info
```

## Project Structure

```
hindi_dubbing/
├── cli/                 # Command-line interface
├── config/              # Configuration management
├── core/
│   ├── audio/           # Audio extraction & processing
│   ├── diarization/     # Speaker diarization (pyannote)
│   ├── transcription/   # Speech-to-text (Whisper)
│   ├── translation/     # Hindi translation (NLLB)
│   ├── voice/           # Voice generation (XTTS-v2)
│   ├── mixing/          # Audio mixing (Demucs)
│   ├── qc/              # Quality control
│   └── render/          # Final video rendering
├── data/
│   └── models.py        # Data models (Pydantic/dataclasses)
├── pipeline/
│   └── orchestrator.py  # Pipeline coordination
└── utils/               # Utilities
```

## Data Models

### ProjectData (Master JSON)
```json
{
  "project_id": "abc123",
  "source_video_path": "movie.mp4",
  "total_duration": 7200.0,
  "speakers": {
    "SPEAKER_01": {
      "character": "Hero",
      "voice_id": "main_SPEAKER_01",
      "voice_type": "main",
      "gender": "male",
      "frequency_percent": 35.2
    }
  },
  "segments": [
    {
      "segment_id": "seg_000001",
      "speaker_id": "SPEAKER_01",
      "character_name": "Hero",
      "time_range": {"start": 10.5, "end": 15.2, "duration": 4.7},
      "original_text": "What are you doing here?",
      "hindi_text": "तुम यहाँ क्या कर रहे हो?",
      "hindi_text_adapted": "तुम यहाँ क्या कर रहे हो?",
      "voice_id": "main_SPEAKER_01",
      "status": "completed"
    }
  ]
}
```

## Key Design Principles

1. **Persistent JSON Dataset**: All analysis saved, no re-processing needed
2. **Chunked Processing**: 10-min chunks with 30s overlap for long movies
3. **Main vs Generic Voices**: Top speakers get dedicated voices, others use pool
4. **Duration Matching**: Hindi dialogue adapted to fit original timing
5. **Selective Regeneration**: Edit individual segments without full re-run
6. **BGM/SFX Preservation**: Demucs source separation keeps background audio

## Requirements

- Python 3.10+
- FFmpeg
- CUDA (recommended for faster inference)
- Hugging Face token (for pyannote diarization)

## Models Used

| Stage | Model | Purpose |
|-------|-------|---------|
| Diarization | pyannote/speaker-diarization-3.1 | Speaker detection |
| Transcription | openai/whisper-large-v3 | Speech-to-text |
| Translation | facebook/nllb-200-distilled-600M | EN→HI translation |
| TTS | coqui/XTTS-v2 | Voice generation |
| Source Sep | htdemucs | BGM/SFX separation |

## License

MIT License - See LICENSE file

## Legal Notice

This tool is for technology development only. Dubbing copyrighted content for distribution requires appropriate licenses. Use with:
- Public domain material
- Properly licensed content
- Content with explicit permission

Technology pipeline ≠ distribution rights.