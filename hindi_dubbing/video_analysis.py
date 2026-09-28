from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path


class VideoAnalyzer:
    """Use Gemini video understanding to map dialogue to visible characters and acting."""

    def __init__(self, settings, log: logging.Logger):
        self.s = settings
        self.log = log

    @staticmethod
    def _plain(obj):
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: VideoAnalyzer._plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [VideoAnalyzer._plain(v) for v in obj]
        if hasattr(obj, 'model_dump'):
            try:
                return VideoAnalyzer._plain(obj.model_dump(mode='json'))
            except Exception:
                pass
        if hasattr(obj, '__dict__'):
            return VideoAnalyzer._plain(vars(obj))
        return str(obj)

    @staticmethod
    def _extract_json(text: str):
        cleaned = text.strip()
        cleaned = re.sub(r'^\s*```(?:json)?\s*', '', cleaned, flags=re.I)
        cleaned = re.sub(r'\s*```\s*$', '', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        obj = re.search(r'\{.*\}', cleaned, re.S)
        arr = re.search(r'\[.*\]', cleaned, re.S)
        candidate = obj.group(0) if obj else (arr.group(0) if arr else None)
        if not candidate:
            raise ValueError('Gemini video response did not contain JSON')
        return json.loads(candidate)

    def _clients(self):
        from google import genai
        for key_no, key in enumerate(self.s.api_keys, 1):
            yield key_no, genai.Client(api_key=key)

    def _upload(self, client, video_path: Path):
        media = client.files.upload(file=str(video_path))
        for _ in range(120):
            state = getattr(media, 'state', None)
            state_name = getattr(state, 'name', str(state) if state else '')
            if state_name in {'ACTIVE', 'SUCCEEDED'}:
                return media
            if state_name in {'FAILED', 'ERROR'}:
                raise RuntimeError(f'Gemini video file processing failed: {state_name}')
            time.sleep(2)
            media = client.files.get(name=media.name)
        raise TimeoutError('Timed out waiting for Gemini video file to become ACTIVE')

    def _analyze_window(self, client, media, window_start, window_end, transcript_segments, registry):
        transcript = '\n'.join(
            f'{s["id"]} | {s["start"]:.2f}-{s["end"]:.2f}s | {s["speaker"]} | {s["text"]}'
            for s in transcript_segments
        )
        registry_text = json.dumps(registry, ensure_ascii=False, indent=2)
        prompt = (
            f'You are the supervising director for a high-quality Hindi anime/movie dub.\n'
            f'Inspect ONLY the video period {window_start:.2f}s to {window_end:.2f}s, but use nearby context when needed.\n'
            'Use the VIDEO: faces, body position, shot composition, mouth/lip movement when visible, relationships, voice context and scene context.\n'
            'Identify the FICTIONAL CHARACTER, not the human voice actor. A female actor can voice a male character and vice versa.\n'
            'Reuse an existing character_id when the same fictional character appears again.\n'
            'Return JSON only with keys characters and annotations.\n'
            'Each annotation must contain segment_id, character_id, confidence, emotion, pace, intensity, style, on_screen.\n'
            'emotion: neutral|happy|sad|angry|fearful|surprised|sarcastic|excited|tired|whisper|other.\n'
            'pace: very_slow|slow|normal|fast|very_fast. intensity: soft|normal|strong|shout.\n'
            'Each character must contain character_id, name, gender, age_group, voice_profile, personality.\n'
            f'Existing character registry:\n{registry_text}\n'
            f'Timed transcript to annotate:\n{transcript}'
        )
        interaction = client.interactions.create(
            model=self.s.video_model,
            input=[
                {'type': 'video', 'uri': media.uri, 'mime_type': getattr(media, 'mime_type', 'video/mp4'), 'processing': 'agentic'},
                {'type': 'text', 'text': prompt},
            ],
        )
        text = getattr(interaction, 'output_text', None)
        if not text:
            text = json.dumps(self._plain(interaction), ensure_ascii=False)
        data = self._extract_json(text)
        if not isinstance(data, dict):
            raise ValueError('Gemini video analysis returned non-object JSON')
        return {'characters': data.get('characters') or [], 'annotations': data.get('annotations') or []}

    def analyze(self, video_path: Path, transcript_segments: list[dict]):
        duration = max((float(s['end']) for s in transcript_segments), default=0.0)
        if duration <= 0:
            return [], {}
        window_len = self.s.video_analysis_window_seconds
        windows = []
        start = 0.0
        while start < duration:
            end = min(duration, start + window_len)
            windows.append((start, end))
            start = end

        last_error = None
        for key_no, client in self._clients():
            media = None
            try:
                self.log.info('Video analysis model=%s key=#%d windows=%d', self.s.video_model, key_no, len(windows))
                media = self._upload(client, video_path)
                annotations = []
                registry = {}
                for window_start, window_end in windows:
                    window_segments = [s for s in transcript_segments if float(s['end']) > window_start and float(s['start']) < window_end]
                    if not window_segments:
                        continue
                    self.log.info('Video analysis window %.2f-%.2fs segments=%d', window_start, window_end, len(window_segments))
                    data = self._analyze_window(client, media, window_start, window_end, window_segments, registry)
                    for character in data['characters']:
                        cid = str(character.get('character_id') or '').strip()
                        if cid:
                            registry.setdefault(cid, {}).update({k: v for k, v in character.items() if v not in (None, '')})
                    annotations.extend([a for a in data['annotations'] if a.get('segment_id')])
                try:
                    client.files.delete(name=media.name)
                except Exception:
                    pass
                return annotations, registry
            except Exception as exc:
                last_error = exc
                self.log.warning('Video analysis key=#%d failed: %s', key_no, str(exc)[:1200])
                if media is not None:
                    try:
                        client.files.delete(name=media.name)
                    except Exception:
                        pass
        raise RuntimeError(f'Gemini video analysis failed on all API keys: {last_error}')