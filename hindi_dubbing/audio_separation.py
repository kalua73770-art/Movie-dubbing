from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

from hindi_dubbing.core.audio.processor import run_cmd


def _resolve_result(item, destination: Path):
    if item is None:
        raise RuntimeError('Stem separation returned an empty file')

    path = getattr(item, 'path', None)
    url = getattr(item, 'url', None)
    if isinstance(item, dict):
        path = path or item.get('path')
        url = url or item.get('url')

    if path and Path(str(path)).exists():
        shutil.copyfile(str(path), destination)
        return destination
    if url:
        urllib.request.urlretrieve(str(url), str(destination))
        return destination
    if isinstance(item, str) and item.startswith(('http://', 'https://')):
        urllib.request.urlretrieve(item, str(destination))
        return destination
    if isinstance(item, str) and Path(item).exists():
        shutil.copyfile(item, destination)
        return destination
    raise RuntimeError(f'Could not resolve separation output: {item}')


def _mix_two(a: Path, b: Path, out: Path, log=None):
    run_cmd([
        'ffmpeg', '-y', '-i', a, '-i', b,
        '-filter_complex',
        '[0:a:0][1:a:0]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,aresample=48000,alimiter=limit=0.98',
        '-ac', '2', '-ar', '48000', '-c:a', 'pcm_s16le', out,
    ], log)


def _concat(files: list[Path], out: Path, log=None):
    list_file = out.parent / f'{out.stem}_concat.txt'
    list_file.write_text(''.join(f"file '{p.as_posix()}'\\n" for p in files), encoding='utf-8')
    run_cmd([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file,
        '-c:a', 'pcm_s16le', '-ac', '2', '-ar', '48000', out,
    ], log)
    try:
        list_file.unlink()
    except Exception:
        pass


def separate_background(source_audio: Path, work_dir: Path, settings, log=None) -> Path | None:
    provider = (settings.audio_stem_provider or 'none').lower().strip()
    if provider in {'none', 'mute', 'off'}:
        return None

    if provider != 'tiger_hf':
        if log:
            log.warning('Unknown AUDIO_STEM_PROVIDER=%s; falling back to exact mute mode', provider)
        return None

    try:
        from gradio_client import Client, handle_file
    except Exception as exc:
        if log:
            log.warning('gradio-client unavailable; falling back to exact mute mode: %s', exc)
        return None

    chunks_dir = work_dir / 'separation_chunks'
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_seconds = int(settings.audio_separation_chunk_seconds)
    duration = None
    try:
        from hindi_dubbing.core.audio.processor import ffprobe_duration
        duration = ffprobe_duration(source_audio)
    except Exception:
        duration = None

    if not duration or duration <= chunk_seconds:
        chunks = [(source_audio, 0.0, duration or 0.0)]
    else:
        chunks = []
        start = 0.0
        idx = 0
        while start < duration:
            end = min(duration, start + chunk_seconds)
            chunk = chunks_dir / f'input_{idx:03d}.wav'
            run_cmd(['ffmpeg', '-y', '-ss', f'{start:.3f}', '-t', f'{end-start:.3f}', '-i', source_audio, '-c:a', 'pcm_s16le', chunk], log)
            chunks.append((chunk, start, end))
            start = end
            idx += 1

    bg_chunks = []
    try:
        client = Client(settings.audio_separation_space)
        for idx, (chunk, _, _) in enumerate(chunks):
            if log:
                log.info('TIGER-DnR separation chunk %d/%d', idx + 1, len(chunks))
            result = client.predict(audio_file=handle_file(str(chunk)), api_name='/separate_dnr')
            if not isinstance(result, (list, tuple)) or len(result) < 3:
                raise RuntimeError(f'Unexpected TIGER-DnR response: {result}')
            effect = chunks_dir / f'effect_{idx:03d}.wav'
            music = chunks_dir / f'music_{idx:03d}.wav'
            bg = chunks_dir / f'background_{idx:03d}.wav'
            _resolve_result(result[1], effect)
            _resolve_result(result[2], music)
            _mix_two(effect, music, bg, log)
            bg_chunks.append(bg)

        final = work_dir / 'audio' / 'background_music_sfx.wav'
        final.parent.mkdir(parents=True, exist_ok=True)
        _concat(bg_chunks, final, log) if len(bg_chunks) > 1 else shutil.copyfile(bg_chunks[0], final)
        if log:
            log.info('TIGER-DnR background stem ready: %s', final)
        return final
    except Exception as exc:
        if log:
            log.warning('TIGER-DnR separation failed; falling back to exact mute: %s', str(exc)[:1200])
        return None