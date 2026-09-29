from __future__ import annotations

import json
import logging
import re
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from hindi_dubbing.audio_separation import separate_background
from hindi_dubbing.core.audio.processor import (
    assemble_track,
    extract_audio,
    extract_range,
    fit_audio,
    trim_edge_silence,
    ffprobe_duration,
    probe_wav,
    run_cmd,
    split_audio,
)
from hindi_dubbing.core.mixing.mixer import mix_final
from hindi_dubbing.gemini import GeminiService
from hindi_dubbing.settings import settings
from hindi_dubbing.video_analysis import VideoAnalyzer

Progress = Callable[[str, str], None]


def _logger(job_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f'dubbing.{job_dir.name}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh = logging.FileHandler(job_dir / 'pipeline.log', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def _state(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def _norm(text: str) -> str:
    return re.sub(r'\W+', '', text.lower(), flags=re.UNICODE)


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _dedupe_segments(segments: list[dict]) -> list[dict]:
    kept = []
    for s in sorted(segments, key=lambda x: (x['start'], x['end'])):
        if not s.get('text', '').strip():
            continue
        duplicate = None
        for i in range(max(0, len(kept) - 20), len(kept)):
            p = kept[i]
            if p['speaker'] != s['speaker']:
                continue
            ov = _overlap(p['start'], p['end'], s['start'], s['end'])
            sim = SequenceMatcher(None, _norm(p['text']), _norm(s['text'])).ratio()
            if ov >= 0.5 or (abs(p['start'] - s['start']) < 0.8 and sim >= 0.72):
                duplicate = i
                break
        if duplicate is None:
            kept.append(dict(s))
        elif len(s['text']) > len(kept[duplicate]['text']):
            kept[duplicate] = dict(s)
    for i, s in enumerate(kept):
        s['id'] = f'seg_{i:06d}'
    return kept


def _batch_by_chars(items: list[dict], limit: int) -> list[list[dict]]:
    batches, current, total = [], [], 0
    for item in items:
        cost = len(item.get('text', '')) + 80
        if current and total + cost > limit:
            batches.append(current)
            current, total = [], 0
        current.append(item)
        total += cost
    if current:
        batches.append(current)
    return batches


def _merge_video_annotations(segments, annotations, registry):
    by_id = {str(a.get('segment_id')): a for a in annotations if a.get('segment_id')}
    for segment in segments:
        ann = by_id.get(segment['id'])
        if ann is None:
            overlaps = [
                a for a in annotations
                if a.get('start') is not None and a.get('end') is not None
                and _overlap(segment['start'], segment['end'], float(a['start']), float(a['end'])) > 0
            ]
            ann = overlaps[0] if overlaps else None

        if ann:
            cid = str(ann.get('character_id') or '').strip()
            if cid:
                segment['character_id'] = cid
            for key in ('emotion', 'pace', 'intensity', 'style'):
                if ann.get(key):
                    segment[key] = ann[key]
        if not segment.get('character_id'):
            segment['character_id'] = f'CHAR_{segment["speaker"]}'

        profile = registry.get(segment['character_id'], {})
        segment['voice_profile'] = profile.get('voice_profile', '')
        segment['character_name'] = profile.get('name', segment['character_id'])

    return segments


def _build_fallback_registry(segments):
    registry = {}
    for s in segments:
        cid = s.get('character_id') or f"CHAR_{s.get('speaker', 'unknown')}"
        s['character_id'] = cid
        registry.setdefault(cid, {
            'character_id': cid,
            'name': cid,
            'gender': 'ambiguous',
            'age_group': 'unknown',
            'voice_profile': '',
            'personality': '',
        })
    return registry


def _make_mix_audio(source_video: Path, out: Path, log):
    out.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        'ffmpeg', '-y', '-i', source_video, '-vn',
        '-ac', '2', '-ar', '48000', '-c:a', 'pcm_s16le', out
    ], log)


def _generate_segment_audio(gemini, segment, voice, segment_dir, log):
    preferred = max(0.25, float(segment['end']) - float(segment['start']))
    next_start = float(segment.get('next_start', segment['start'] + preferred))
    available = max(preferred, next_start - float(segment['start']) - 0.06)

    # Allow the dubbed line to use nearby silence rather than forcing the TTS
    # to exactly match a very short English word window.
    natural_cap = min(available, preferred + 1.50)
    attempts = max(0, min(int(settings.rewrite_attempts), 1))

    best_path = None
    best_duration = None
    for attempt in range(attempts + 1):
        raw = segment_dir / f'{segment["id"]}_attempt{attempt}.wav'
        gemini.tts_segment(segment, voice, raw)

        trimmed = segment_dir / f'{segment["id"]}_attempt{attempt}_trim.wav'
        trim_edge_silence(raw, trimmed, log)
        observed = probe_wav(trimmed)[3]
        ratio = observed / natural_cap if natural_cap else 1.0

        log.info(
            'Dialogue %s preferred=%.3fs natural_cap=%.3fs available=%.3fs '
            'generated=%.3fs ratio_to_cap=%.3f attempt=%d',
            segment['id'], preferred, natural_cap, available, observed, ratio, attempt,
        )

        if best_duration is None or abs(observed - preferred) < abs(best_duration - preferred):
            best_path = trimmed
            best_duration = observed

        if observed <= natural_cap:
            segment['tts_duration'] = observed
            segment['dub_end'] = float(segment['start']) + observed
            segment['tts_time_stretch'] = 1.0
            return trimmed

        if attempt < attempts:
            segment['hindi'] = gemini.rewrite_for_observed_duration(
                segment,
                observed,
                target_duration=natural_cap,
            )

    # Never abort the whole movie because one very short line does not fit
    # perfectly. Prefer a mild speed adjustment over an extreme stretch.
    fit = segment_dir / f'{segment["id"]}.wav'
    fit_audio(best_path, fit, natural_cap, log)
    fitted = probe_wav(fit)[3]
    segment['tts_duration'] = fitted
    segment['dub_end'] = float(segment['start']) + fitted
    segment['tts_time_stretch'] = best_duration / natural_cap if natural_cap else 1.0
    if best_duration > natural_cap:
        log.warning(
            'Dialogue %s required mild timing correction: %.3fs -> %.3fs',
            segment['id'], best_duration, natural_cap,
        )
    return fit

def run_pipeline(source_video: Path, output_video: Path, progress: Progress | None = None, job_id: str | None = None) -> Path:
    settings.validate()
    job_id = job_id or uuid.uuid4().hex[:12]
    work = Path(settings.work_dir) / 'jobs' / job_id
    chunks_dir = work / 'chunks'
    audio_dir = work / 'audio'
    segments_dir = work / 'segments'
    for directory in (chunks_dir, audio_dir, segments_dir):
        directory.mkdir(parents=True, exist_ok=True)

    log = _logger(work)
    manifest = work / 'project.json'
    state = {
        'job_id': job_id,
        'status': 'running',
        'stage': 'starting',
        'source': str(source_video),
        'output': str(output_video),
        'segments': [],
        'characters': {},
        'models': {},
    }
    _state(manifest, state)

    def report(stage, message):
        state['stage'] = stage
        state['message'] = message
        _state(manifest, state)
        if progress:
            try:
                progress(stage, message)
            except Exception:
                pass
        log.info('%s: %s', stage, message)

    try:
        report('audio', 'Extracting precision transcription and mix audio')
        source_audio = audio_dir / 'source_16k.wav'
        mix_audio = audio_dir / 'source_48k.wav'
        extract_audio(source_video, source_audio, log)
        _make_mix_audio(source_video, mix_audio, log)
        duration = ffprobe_duration(source_video)

        chunks = split_audio(
            source_audio, chunks_dir, duration,
            settings.transcribe_chunk_seconds,
            settings.transcribe_overlap_seconds,
            log,
        )
        gemini = GeminiService(settings, log)
        state['models']['transcribe'] = settings.transcribe_model
        state['models']['video'] = settings.video_models
        state['models']['tts'] = settings.tts_models

        all_segments = []
        for idx, (chunk, start, end) in enumerate(chunks):
            report('transcription', f'Transcribing chunk {idx + 1}/{len(chunks)}')
            local = gemini.transcribe(chunk)
            for segment in local:
                item = dict(segment)
                item['start'] += start
                item['end'] += start
                all_segments.append(item)

        segments = _dedupe_segments(all_segments)
        if not segments:
            raise RuntimeError('No speech segments returned by Gemini Transcribe')

        report('video', 'Using Gemini video understanding for visible character and acting analysis')
        registry = {}
        try:
            annotations, registry = VideoAnalyzer(settings, log).analyze(source_video, segments)
            _merge_video_annotations(segments, annotations, registry)
        except Exception as exc:
            log.warning('Video analysis failed; using audio diarization fallback: %s', str(exc)[:1200])
            registry = _build_fallback_registry(segments)
            _merge_video_annotations(segments, [], registry)

        fallback_registry = _build_fallback_registry(segments)
        for cid, fallback in fallback_registry.items():
            registry.setdefault(cid, fallback)
        voice_map = gemini.choose_voices(registry)
        for segment in segments:
            segment['voice'] = voice_map[segment['character_id']]

        state['characters'] = {
            cid: {**registry.get(cid, {}), 'voice': voice_map[cid]}
            for cid in voice_map
        }
        report('video', f'Locked {len(voice_map)} character voices and acting profiles')

        report('translation', 'Generating duration-aware Hindi dialogue')
        for batch_no, batch in enumerate(_batch_by_chars(segments, settings.translation_batch_chars), 1):
            report('translation', f'Translation batch {batch_no}')
            translated = gemini.translate_for_duration(batch)
            for segment in batch:
                segment['hindi'] = translated[segment['id']]

        audio_items = []
        ordered_segments = sorted(segments, key=lambda x: float(x['start']))
        for i, segment in enumerate(ordered_segments):
            segment['next_start'] = float(ordered_segments[i + 1]['start']) if i + 1 < len(ordered_segments) else duration
        report('tts', 'Synthesizing each dialogue line with locked character voice and acting style')
        for idx, segment in enumerate(segments, 1):
            report('tts', f'Dialogue {idx}/{len(segments)}')
            audio_path = _generate_segment_audio(
                gemini,
                segment,
                voice_map[segment['character_id']],
                segments_dir,
                log,
            )
            segment['audio_path'] = str(audio_path)
            audio_items.append((segment['start'], segment['dub_end'], audio_path))

        report('mix', 'Building Hindi dialogue timeline')
        dialogue_track = audio_dir / 'hindi_dialogue.wav'
        assemble_track(audio_items, duration, dialogue_track)

        report('separation', f'Separating dialogue/music/effects via {settings.audio_stem_provider}')
        background = separate_background(mix_audio, work, settings, log)

        report('render', 'Rendering final movie with background/SFX preserved when available')
        output_video.parent.mkdir(parents=True, exist_ok=True)
        mix_final(
            source_video,
            dialogue_track,
            background,
            output_video,
            segments=segments,
            log=log,
        )

        state['segments'] = segments
        state['status'] = 'completed'
        state['stage'] = 'done'
        _state(manifest, state)
        report('done', 'High-quality Hindi dub is ready')
        return output_video
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = str(exc)
        _state(manifest, state)
        log.exception('Pipeline failed')
        if progress:
            try:
                progress('error', str(exc))
            except Exception:
                pass
        raise