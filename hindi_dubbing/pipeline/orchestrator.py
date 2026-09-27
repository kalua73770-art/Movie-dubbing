from __future__ import annotations

import json
import logging
import re
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from hindi_dubbing.core.audio.processor import (
    assemble_track,
    extract_audio,
    extract_range,
    ffprobe_duration,
    fit_audio,
    silence_ranges,
    split_audio,
)
from hindi_dubbing.core.mixing.mixer import mix_and_duck
from hindi_dubbing.gemini import GeminiService
from hindi_dubbing.settings import settings

Progress = Callable[[str, str], None]


def _logger(job_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"dubbing.{job_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh = logging.FileHandler(job_dir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def _state(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm(text: str) -> str:
    return re.sub(r"\W+", "", text.lower(), flags=re.UNICODE)


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _reconcile_speakers(local_segments: list[dict], chunk_start: float, previous: list[dict], next_id: list[int]) -> list[dict]:
    mapping: dict[str, str] = {}
    local_ids = list(dict.fromkeys(s["speaker"] for s in local_segments))
    overlap_start = chunk_start - 1.0
    overlap_end = chunk_start + settings.transcribe_overlap_seconds + 1.0

    for lid in local_ids:
        candidates: dict[str, float] = {}
        for p in previous:
            if p["end"] < overlap_start or p["start"] > overlap_end:
                continue
            for s in local_segments:
                if s["speaker"] != lid:
                    continue
                a0, a1 = chunk_start + float(s["start"]), chunk_start + float(s["end"])
                candidates[p["speaker"]] = candidates.get(p["speaker"], 0.0) + _overlap(
                    a0, a1, p["start"], p["end"]
                )
        if candidates:
            best, score = max(candidates.items(), key=lambda kv: kv[1])
            if score >= 0.15:
                mapping[lid] = best
        if lid not in mapping:
            mapping[lid] = f"spk_{next_id[0]}"
            next_id[0] += 1

    out = []
    for s in local_segments:
        x = dict(s)
        x["speaker"] = mapping[x["speaker"]]
        x["start"] = chunk_start + float(s["start"])
        x["end"] = chunk_start + float(s["end"])
        out.append(x)
    return out


def _dedupe_segments(segments: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for s in sorted(segments, key=lambda x: (x["start"], x["end"])):
        if not s.get("text", "").strip():
            continue
        duplicate = None
        for i in range(max(0, len(kept) - 15), len(kept)):
            p = kept[i]
            if p["speaker"] != s["speaker"]:
                continue
            overlap = _overlap(p["start"], p["end"], s["start"], s["end"])
            sim = SequenceMatcher(None, _norm(p["text"]), _norm(s["text"])).ratio()
            if overlap >= 0.25 or (abs(p["start"] - s["start"]) < 0.9 and sim >= 0.70):
                if sim >= 0.65 or overlap >= 0.80:
                    duplicate = i
                    break
        if duplicate is None:
            kept.append(s)
        elif len(s["text"]) > len(kept[duplicate]["text"]):
            kept[duplicate] = s

    for i, s in enumerate(kept):
        s["id"] = f"seg_{i:06d}"
    return kept


def _batch_by_chars(items: list[dict], limit: int) -> list[list[dict]]:
    batches: list[list[dict]] = []
    cur: list[dict] = []
    total = 0
    for item in items:
        cost = len(item.get("text", "")) + 32
        if cur and total + cost > limit:
            batches.append(cur)
            cur, total = [], 0
        cur.append(item)
        total += cost
    if cur:
        batches.append(cur)
    return batches


def _tts_batches(segments: list[dict], max_seconds: int, max_chars: int) -> list[list[dict]]:
    batches: list[list[dict]] = []
    cur: list[dict] = []
    cur_start = cur_end = None
    chars = 0
    speakers: set[str] = set()

    for s in segments:
        n_chars = len(s.get("hindi", "")) + 16
        if cur:
            span = max(cur_end or s["end"], s["end"]) - min(cur_start or s["start"], s["start"])
            too_big = span > max_seconds or chars + n_chars > max_chars
            too_many_speakers = len(speakers | {s["speaker"]}) > 2
            if too_big or too_many_speakers:
                batches.append(cur)
                cur, cur_start, cur_end, chars, speakers = [], None, None, 0, set()
        cur.append(s)
        cur_start = s["start"] if cur_start is None else min(cur_start, s["start"])
        cur_end = s["end"] if cur_end is None else max(cur_end, s["end"])
        chars += n_chars
        speakers.add(s["speaker"])

    if cur:
        batches.append(cur)
    return batches


def _build_speaker_sample(source_audio: Path, speaker_segments: list[dict], out_dir: Path, speaker: str, log: logging.Logger) -> Path:
    """Build a short concatenated audio sample for character/voice analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    cursor = 0.0

    for idx, segment in enumerate(sorted(speaker_segments, key=lambda x: x["start"])):
        duration = min(3.5, max(0.0, float(segment["end"]) - float(segment["start"])))
        if duration < 0.35:
            continue

        raw = out_dir / f"{speaker}_{idx:02d}.wav"
        extract_range(
            source_audio,
            raw,
            float(segment["start"]),
            float(segment["start"]) + duration,
            log,
        )
        items.append((cursor, cursor + duration, raw))
        cursor += duration + 0.15

        if cursor >= 8.0:
            break

    if not items:
        raise RuntimeError(f"No usable audio found for speaker {speaker}")

    sample = out_dir / f"{speaker}_sample.wav"
    assemble_track(items, cursor, sample)
    return sample


def _partition_ranges(ranges: list[tuple[float, float]], target_durations: list[float]) -> list[list[tuple[float, float]]]:
    if not ranges or not target_durations or len(ranges) < len(target_durations):
        return [[] for _ in target_durations]

    n, g = len(ranges), len(target_durations)
    prefix = [0.0]
    for a, b in ranges:
        prefix.append(prefix[-1] + max(0.0, b - a))

    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(g + 1)]
    parent = [[-1] * (n + 1) for _ in range(g + 1)]
    dp[0][0] = 0.0

    for groups in range(1, g + 1):
        for j in range(groups, n + 1):
            for k in range(groups - 1, j):
                dur = prefix[j] - prefix[k]
                target = target_durations[groups - 1]
                value = dp[groups - 1][k] + (dur - target) ** 2
                if value < dp[groups][j]:
                    dp[groups][j] = value
                    parent[groups][j] = k

    if parent[g][n] < 0:
        return [[] for _ in target_durations]

    groups: list[list[tuple[float, float]]] = []
    j = n
    for gi in range(g, 0, -1):
        k = parent[gi][j]
        groups.append(ranges[k:j])
        j = k
    groups.reverse()
    return groups


def _assign_batch_audio(batch, generated: Path, base_dir: Path, log: logging.Logger, batch_no: int):
    ranges = silence_ranges(generated, log=log)
    if not ranges:
        from hindi_dubbing.core.audio.processor import probe_wav
        ranges = [(0.0, probe_wav(generated)[3])]

    targets = [max(0.15, s["end"] - s["start"]) for s in batch]
    groups = _partition_ranges(ranges, targets)

    if any(not g for g in groups):
        from hindi_dubbing.core.audio.processor import probe_wav
        total_audio = probe_wav(generated)[3]
        total_target = sum(targets)
        groups = []
        cursor = 0.0
        for target in targets:
            piece = total_audio * (target / total_target)
            groups.append([(cursor, min(total_audio, cursor + piece))])
            cursor += piece

    paths = []
    for i, (segment, rg) in enumerate(zip(batch, groups)):
        start, end = rg[0][0], rg[-1][1]
        raw = base_dir / f"batch_{batch_no:04d}_{i:02d}_raw.wav"
        fit = base_dir / f"batch_{batch_no:04d}_{i:02d}.wav"
        extract_range(generated, raw, start, end, log)
        fit_audio(raw, fit, max(0.12, segment["end"] - segment["start"]), log)
        paths.append(fit)
    return paths


def run_pipeline(source_video: Path, output_video: Path, progress: Progress | None = None, job_id: str | None = None) -> Path:
    settings.validate()
    job_id = job_id or uuid.uuid4().hex[:12]
    work = Path(settings.work_dir) / "jobs" / job_id
    chunks_dir, audio_dir = work / "chunks", work / "audio"
    tts_dir, segments_dir = work / "tts", work / "segments"
    for p in (chunks_dir, audio_dir, tts_dir, segments_dir):
        p.mkdir(parents=True, exist_ok=True)

    log = _logger(work)
    manifest = work / "project.json"
    state = {
        "job_id": job_id, "status": "running", "stage": "starting",
        "source": str(source_video), "output": str(output_video),
        "segments": [], "speakers": {}, "models": {}
    }
    _state(manifest, state)

    def report(stage: str, message: str):
        state["stage"], state["message"] = stage, message
        _state(manifest, state)
        if progress:
            try:
                progress(stage, message)
            except Exception:
                pass
        log.info("%s: %s", stage, message)

    try:
        report("audio", "Extracting source audio with FFmpeg")
        source_audio = audio_dir / "source.wav"
        extract_audio(source_video, source_audio, log)
        duration = ffprobe_duration(source_video)
        chunks = split_audio(
            source_audio, chunks_dir, duration,
            settings.transcribe_chunk_seconds,
            settings.transcribe_overlap_seconds, log
        )
        report("audio", f"Prepared {len(chunks)} transcription chunk(s)")

        gemini = GeminiService(settings, log)
        all_segments: list[dict] = []
        next_speaker = [1]

        for idx, (chunk, start, end) in enumerate(chunks):
            report("transcription", f"Transcribing chunk {idx + 1}/{len(chunks)}")
            local = gemini.transcribe(chunk)
            mapped = _reconcile_speakers(local, start, all_segments, next_speaker)
            all_segments.extend(mapped)
        state["models"]["transcribe"] = settings.transcribe_model

        segments = _dedupe_segments(all_segments)
        if not segments:
            raise RuntimeError("No speech segments returned by Gemini Transcribe")

        speakers = sorted(set(s["speaker"] for s in segments), key=lambda x: x)
        report("transcription", f"Found {len(segments)} dialogue segments / {len(speakers)} speakers")

        report("speakers", "Analyzing character voices and selecting gender-aware TTS voices")
        speaker_samples_dir = work / "speakers"
        speaker_profiles = {}
        speaker_segments = {
            sp: [s for s in segments if s["speaker"] == sp]
            for sp in speakers
        }
        for sp in speakers:
            sample = _build_speaker_sample(
                source_audio,
                speaker_segments[sp],
                speaker_samples_dir,
                sp,
                log,
            )
            context = " ".join(s["text"] for s in speaker_segments[sp][:12])
            speaker_profiles[sp] = gemini.classify_speaker(sample, context)
            try:
                sample.unlink()
            except Exception:
                pass

        voice_map = gemini.choose_voices(speaker_profiles)
        state["speakers"] = {
            sp: {
                **speaker_profiles[sp],
                "voice": voice_map[sp],
            }
            for sp in speakers
        }

        report("translation", "Translating dialogue with Gemini Flash-Lite")
        for no, batch in enumerate(_batch_by_chars(segments, settings.translation_batch_chars), 1):
            report("translation", f"Translation batch {no}")
            mapping = gemini.translate_batch(batch)
            for s in batch:
                s["hindi"] = mapping[s["id"]]
        state["models"]["translation"] = settings.text_models

        report("tts", "Generating Hindi speech in large batches")
        audio_items: list[tuple[float, float, Path]] = []
        batches = _tts_batches(segments, settings.tts_batch_seconds, settings.tts_batch_chars)

        for bno, batch in enumerate(batches, 1):
            report("tts", f"TTS batch {bno}/{len(batches)}")
            batch_audio = tts_dir / f"batch_{bno:04d}.wav"
            gemini.tts(batch, voice_map, batch_audio)
            state["models"]["tts"] = settings.tts_models
            mapped_audio = _assign_batch_audio(batch, batch_audio, segments_dir, log, bno)
            if len(mapped_audio) != len(batch):
                raise RuntimeError("Could not map generated batch audio back to dialogue segments")
            for s, audio in zip(batch, mapped_audio):
                s["audio_path"] = str(audio)
                audio_items.append((s["start"], s["end"], audio))

        report("mix", "Building Hindi dialogue timeline")
        dialogue_track = audio_dir / "hindi_dialogue.wav"
        assemble_track(audio_items, duration, dialogue_track)

        report("render", "Ducking original audio and rendering final movie")
        output_video.parent.mkdir(parents=True, exist_ok=True)
        mix_and_duck(source_video, dialogue_track, output_video, settings.duck_db, log)

        state["segments"] = segments
        state["status"], state["stage"] = "completed", "done"
        _state(manifest, state)
        report("done", "Hindi dubbed movie is ready")
        return output_video
    except Exception as exc:
        state["status"], state["error"] = "failed", str(exc)
        _state(manifest, state)
        log.exception("Pipeline failed")
        if progress:
            try:
                progress("error", str(exc))
            except Exception:
                pass
        raise
