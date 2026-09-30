import logging
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from hindi_dubbing.gemini import GeminiService
from hindi_dubbing.settings import Settings
from hindi_dubbing.video_analysis import VideoAnalyzer


class FakeModels:
    def __init__(self, key):
        self.key = key

    def generate_content(self, model, contents, config=None):
        if self.key == "bad-key":
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")
        if self.key == "empty-key":
            return SimpleNamespace(text="")
        if model == "bad-model":
            raise RuntimeError("503 service unavailable")
        return SimpleNamespace(text="successful response")


class FakeClient:
    def __init__(self, api_key, **kwargs):
        self.api_key = api_key
        self.models = FakeModels(api_key)


class ResilienceTests(unittest.TestCase):
    def make_settings(self, **overrides):
        values = dict(
            api_keys=["bad-key", "good-key"],
            text_models=["text-model"],
            tts_models=["tts-model"],
            transcribe_model="gemini-3.5-transcribe",
            video_models=["video-model"],
            transcribe_chunk_seconds=900,
            transcribe_overlap_seconds=8,
            transcribe_workers=2,
            video_analysis_max_windows=6,
            gemini_http_timeout_ms=600000,
            text_timeout_ms=120000,
            transcribe_timeout_ms=600000,
            tts_timeout_ms=180000,
            video_timeout_ms=600000,
            speaker_analysis_timeout_ms=120000,
        )
        values.update(overrides)
        return Settings(**values)

    def test_text_falls_back_to_next_key_after_quota(self):
        settings = self.make_settings()
        service = GeminiService(settings, logging.getLogger("test"))
        with patch("hindi_dubbing.gemini.genai.Client", side_effect=FakeClient):
            result, model = service.generate_text("hello")
        self.assertEqual(result, "successful response")
        self.assertEqual(model, "text-model")

    def test_empty_response_falls_back_to_next_key(self):
        settings = self.make_settings(api_keys=["empty-key", "good-key"])
        service = GeminiService(settings, logging.getLogger("test"))
        with patch("hindi_dubbing.gemini.genai.Client", side_effect=FakeClient):
            result, model = service.generate_text("hello")
        self.assertEqual(result, "successful response")
        self.assertEqual(model, "text-model")

    def test_text_model_pool_falls_back_to_next_model(self):
        settings = self.make_settings(
            api_keys=["good-key"],
            text_models=["bad-model", "good-model"],
        )
        service = GeminiService(settings, logging.getLogger("test"))
        with patch("hindi_dubbing.gemini.genai.Client", side_effect=FakeClient):
            result, model = service.generate_text("hello")
        self.assertEqual(result, "successful response")
        self.assertEqual(model, "good-model")

    def test_video_window_cap_is_deterministic(self):
        windows = [(i * 10.0, i * 10.0 + 10.0) for i in range(12)]
        selected = VideoAnalyzer._select_windows(windows, 6)
        self.assertEqual(len(selected), 6)
        self.assertEqual(selected[0], windows[0])
        self.assertEqual(selected[-1], windows[-1])

    def test_settings_reject_invalid_chunk_configuration(self):
        settings = self.make_settings(transcribe_chunk_seconds=5, transcribe_overlap_seconds=5)
        with self.assertRaises(RuntimeError):
            settings.validate()


if __name__ == "__main__":
    unittest.main()
