import logging
import tempfile
import unittest
from pathlib import Path

from hindi_dubbing.gemini import GeminiService
from hindi_dubbing.movie_brain import MovieBrain
from hindi_dubbing.settings import Settings


class ContinuityTests(unittest.TestCase):
    def make_settings(self):
        return Settings(
            api_keys=["k1", "k2"],
            text_models=["text-model"],
            tts_models=[
                "gemini-3.8-flash-tts",
                "gemini-3.8-flash-lite-tts",
                "gemini-3.1-flash-tts-preview",
                "gemini-2.5-flash-preview-tts",
            ],
            transcribe_model="gemini-3.5-transcribe",
            video_models=["video-model"],
            transcribe_chunk_seconds=900,
            transcribe_overlap_seconds=8,
            transcribe_workers=2,
            translation_workers=2,
            tts_workers=2,
            video_analysis_max_windows=6,
            gemini_http_timeout_ms=60000,
            text_timeout_ms=45000,
            transcribe_timeout_ms=60000,
            tts_timeout_ms=50000,
            video_timeout_ms=60000,
            speaker_analysis_timeout_ms=60000,
        )

    def test_multi_speaker_tts_payload(self):
        service = GeminiService(self.make_settings(), logging.getLogger("test-tts"))
        segments = [
            {"id": "a", "speaker": "spk1", "hindi": "hello"},
            {"id": "b", "speaker": "spk2", "hindi": "world"},
        ]
        payload, config = service.build_tts_input(
            segments,
            {"spk1": "Puck", "spk2": "Kore"},
        )
        content = payload[0]["content"]
        self.assertEqual(content[0]["annotations"][0]["speaker"], "spk1")
        self.assertEqual(content[1]["annotations"][0]["speaker"], "spk2")
        self.assertEqual(len(config["speakers"]), 2)

    def test_movie_brain_long_context_and_voice_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MovieBrain(Path(tmp) / "movie_brain.json").load()
            brain.merge_characters({
                "c1": {
                    "character_id": "c1",
                    "name": "Hero",
                    "gender": "male",
                }
            })
            brain.lock_voices({"c1": "Puck"})
            brain.lock_voices({"c1": "Kore"})
            self.assertEqual(brain.data["voice_locks"]["c1"], "Puck")
            for i in range(20):
                brain.add_scene_summary(i * 10, i * 10 + 10, "scene " + ("x" * 500))
            brain.save()
            self.assertGreater(len(brain.agent_context()), 6500)


if __name__ == "__main__":
    unittest.main()
