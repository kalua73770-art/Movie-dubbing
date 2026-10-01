from __future__ import annotations

import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from hindi_dubbing.background_reconstruction import reconstruct_background
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
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _norm(text: str) -> str:
    return re.sub(r"\W+", "", text.lower(), flags=re.UNICODE)


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _dedupe_segments(segments: list[dict]) -> list[dict]:
    kept = []
    for s in sorted(segments, key=lambda x: (x["start"], x["end"])):
        if not s.get("text", "").strip():
            continue
        duplicate = None
        for i in range(max(0, len(kept) - 20), len(kept)):
            p = kept[i]
            if p["speaker"] != s["speaker"]:
                continue
            ov = _overlap(p["start"], p["end"], s["start"], s["end"])
            sim = SequenceMatcher(
                None,
                _norm(p["text"]),
                _norm(s["text"]),
            ).ratio()
            if ov >= 0.5 or (
                abs(p["start"] - s["start"]) < 0.8 and sim >= 0.72
            ):
                duplicate = i
                break
        if duplicate is None:
            kept.append(dict(s))
        elif len(s["text"]) > len(kept[duplicate]["text"]):
            kept[duplicate] = dict(s)
    for i, s in enumerate(kept):
        s["id"] = f"seg_{i:06d}"
    return kept


def _batch_by_chars(items: list[dict], limit: int) -> list[list[dict]]:
    batches, current, total = [], [], 0
    for item in items:
        cost = len(item.get("text", "")) + 80
        if current and total + cost > limit:
            batches.append(current)
            current, total = [], 0
        current.append(item)
        total += cost
    if current:
        batches.append(current)
    return batches


def _merge_video_annotations(segments, annotations, registry):
    by_id = {
        str(a.get("segment_id")): a
        for a in annotations
        if a.get("segment_id")
    }

    # Build a speaker -> character map from the segment annotations returned by
    # video understanding. This prevents the same person from becoming multiple
    # fallback characters whenever a particular frame is hard to attribute.
    speaker_votes = {}
    segment_lookup = {s["id"]: s for s in segments}
    for ann in annotations:
        seg = segment_lookup.get(str(ann.get("segment_id")))
        cid = str(ann.get("character_id") or "").strip()
        if not seg or not cid:
            continue
        speaker = seg.get("speaker")
        if not speaker:
            continue
        try:
            confidence = float(ann.get("confidence", 0.5) or 0.5)
        except Exception:
            confidence = 0.5
        speaker_votes.setdefault(speaker, {}).setdefault(cid, 0.0)
        speaker_votes[speaker][cid] += max(0.1, confidence)

    dominant_character = {
        speaker: max(votes.items(), key=lambda kv: kv[1])[0]
        for speaker, votes in speaker_votes.items()
        if votes
    }

    for segment in segments:
        ann = by_id.get(segment["id"])
        if ann is None:
            overlaps = [
                a
                for a in annotations
                if a.get("start") is not None
                and a.get("end") is not None
                and _overlap(
                    segment["start"],
                    segment["end"],
                    float(a["start"]),
                    float(a["end"]),
                ) > 0
            ]
            ann = overlaps[0] if overlaps else None

        if ann:
            cid = str(ann.get("character_id") or "").strip()
            if cid:
                segment["character_id"] = cid
            for key in ("emotion", "pace", "intensity", "style"):
                if ann.get(key):
                    segment[key] = ann[key]

        # If this exact line was not visually attributed, reuse the dominant
        # character for the same diarized speaker instead of creating a new voice.
        if not segment.get("character_id"):
            segment["character_id"] = dominant_character.get(
                segment.get("speaker"),
                f'CHAR_{segment["speaker"]}',
            )

        profile = registry.get(segment["character_id"], {})
        segment["voice_profile"] = profile.get("voice_profile", "")
        segment["character_name"] = profile.get(
            "name",
            segment["character_id"],
        )

    return segments


def _build_fallback_registry(segments):
    registry = {}
    for s in segments:
        cid = s.get("character_id") or f'CHAR_{s.get("speaker", "unknown")}'
        s["character_id"] = cid
        registry.setdefault(
            cid,
            {
                "character_id": cid,
                "name": cid,
                "gender": "ambiguous",
                "age_group": "unknown",
                "voice_profile": "",
                "personality": "",
            },
        )
    return registry


def _make_mix_audio(source_video: Path, out: Path, log):
    out.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            source_video,
            "-vn",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            out,
        ],
        log,
    )


def _generate_segment_audio(gemini, segment, voice, segment_dir, log):
    preferred = max(
        0.25,
        float(segment["end"]) - float(segment["start"]),
    )
    next_start = float(
        segment.get("next_start", segment["start"] + preferred)
    )
    available = max(
        preferred,
        next_start - float(segment["start"]) - 0.06,
    )

    natural_min = max(0.30, preferred * 0.82)
    natural_cap = min(available, max(preferred * 1.18, preferred + 1.0))
    attempts = max(0, min(int(settings.rewrite_attempts), 1))

    best_path = None
    best_distance = float("inf")
    best_duration = 0.0

    for attempt in range(attempts + 1):
        raw = segment_dir / f'{segment["id"]}_attempt{attempt}.wav'
        gemini.tts_segment(segment, voice, raw)

        trimmed = segment_dir / f'{segment["id"]}_attempt{attempt}_trim.wav'
        trim_edge_silence(raw, trimmed, log)
        observed = probe_wav(trimmed)[3]

        if natural_min <= observed <= natural_cap:
            segment["tts_duration"] = observed
            segment["dub_end"] = min(
                float(segment["start"]) + observed,
                next_start - 0.02,
            )
            segment["tts_time_stretch"] = 1.0
            return trimmed

        distance = 0.0
        if observed < natural_min:
            distance = natural_min - observed
        elif observed > natural_cap:
            distance = observed - natural_cap

        if distance < best_distance:
            best_path = trimmed
            best_distance = distance
            best_duration = observed

        log.info(
            "Dialogue %s preferred=%.3fs min=%.3fs cap=%.3fs "
            "available=%.3fs generated=%.3fs attempt=%d",
            segment["id"],
            preferred,
            natural_min,
            natural_cap,
            available,
            observed,
            attempt,
        )

        if attempt < attempts:
            segment["hindi"] = gemini.rewrite_for_observed_duration(
                segment,
                observed,
                target_duration=(
                    natural_min if observed < natural_min else natural_cap
                ),
            )

    # Last resort: only mild timing correction. The linguistic rewrite above has
    # already tried to pull the line into a natural range.
    target = natural_min if best_duration < natural_min else natural_cap
    fit = segment_dir / f'{segment["id"]}.wav'
    ratio = best_duration / target if target else 1.0

    if 0.60 <= ratio <= 1.60:
        fit_audio(best_path, fit, target, log)
        fitted = probe_wav(fit)[3]
        segment["tts_duration"] = fitted
        segment["dub_end"] = min(
            float(segment["start"]) + fitted,
            next_start - 0.02,
        )
        segment["tts_time_stretch"] = ratio
        log.warning(
            "Dialogue %s required mild timing correction: %.3fs -> %.3fs",
            segment["id"],
            best_duration,
            target,
        )
        return fit

    raise RuntimeError(
        f"Could not produce usable TTS for {segment['id']}: "
        f"{best_duration:.2f}s target={target:.2f}s"
    )

def run_pipeline(
    source_video: Path,
    output_video: Path,
    progress: Progress | None = None,
    job_id: str | None = None,
) -> Path:
    settings.validate()
    job_id = job_id or uuid.uuid4().hex[:12]
    work = Path(settings.work_dir) / "jobs" / job_id
    chunks_dir = work / "chunks"
    audio_dir = work / "audio"
    segments_dir = work / "segments"

    for directory in (chunks_dir, audio_dir, segments_dir):
        directory.mkdir(parents=True, exist_ok=True)

    log = _logger(work)
    manifest = work / "project.json"
    state = {
        "job_id": job_id,
        "status": "running",
        "stage": "starting",
        "source": str(source_video),
        "output": str(output_video),
        "segments": [],
        "characters": {},
        "models": {},
    }
    _state(manifest, state)

    def report(stage, message):
        state["stage"] = stage
        state["message"] = message
        _state(manifest, state)
        if progress:
            try:
                progress(stage, message)
            except Exception:
                pass
        log.info("%s: %s", stage, message)

    try:
        report(
            "audio",
            "Extracting precision transcription and mix audio",
        )
        source_audio = audio_dir / "source_16k.wav"
        mix_audio = audio_dir / "source_48k.wav"
        extract_audio(source_video, source_audio, log)
        _make_mix_audio(source_video, mix_audio, log)
        duration = ffprobe_duration(source_video)

        chunks = split_audio(
            source_audio,
            chunks_dir,
            duration,
            settings.transcribe_chunk_seconds,
            settings.transcribe_overlap_seconds,
            log,
        )
        gemini = GeminiService(settings, log)

        report("model_check", "Checking configured Gemini model endpoints before expensive work")
        health = gemini.probe_models({
            "text": settings.text_models,
            "video": settings.video_models,
            "tts": settings.tts_models,
            "transcribe": [settings.transcribe_model],
        })

        if health.get("text"):
            settings.text_models[:] = health["text"]
        if health.get("video"):
            settings.video_models[:] = health["video"]
        if health.get("tts"):
            settings.tts_models[:] = health["tts"]

        # Transcribe has a dedicated model; runtime key rotation remains enabled.
        if not health.get("transcribe"):
            raise RuntimeError("Gemini Transcribe model is not available to any configured API key")

        if not settings.text_models:
            raise RuntimeError("No preflight-ready Gemini text model is available")
        if not settings.tts_models:
            raise RuntimeError("No preflight-ready Gemini TTS model is available")

        state["models"]["transcribe"] = settings.transcribe_model
        state["models"]["video"] = settings.video_models
        state["models"]["tts"] = settings.tts_models
        state["models"]["preflight"] = health

        # Transcription chunks are independent. When multiple API keys are
        # configured, use bounded parallelism and pin initial chunks to different
        # keys so one busy key does not stall the entire movie.
        workers = min(
            max(1, int(settings.transcribe_workers)),
            max(1, len(settings.api_keys)),
            max(1, len(chunks)),
        )
        all_segments = [None] * len(chunks)

        def transcribe_one(idx, chunk_info):
            chunk, start, end = chunk_info
            preferred_key_index = idx % max(1, len(settings.api_keys))
            local = gemini.transcribe(
                chunk,
                preferred_key_index=preferred_key_index,
            )
            return idx, start, end, local

        report(
            "transcription",
            f"Transcribing {len(chunks)} chunks with {workers} worker(s)",
        )

        if workers == 1:
            completed = 0
            for idx, chunk_info in enumerate(chunks):
                result = transcribe_one(idx, chunk_info)
                all_segments[idx] = result
                completed += 1
                report(
                    "transcription",
                    f"Finished chunk {completed}/{len(chunks)}",
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(transcribe_one, idx, chunk_info): idx
                    for idx, chunk_info in enumerate(chunks)
                }
                completed = 0
                for future in as_completed(future_map):
                    result = future.result()
                    idx, start, end, local = result
                    all_segments[idx] = result
                    completed += 1
                    report(
                        "transcription",
                        f"Finished chunk {completed}/{len(chunks)} "
                        f"(chunk {idx + 1})",
                    )

        merged = []
        for item in all_segments:
            _, start, _, local = item
            for segment in local:
                shifted = dict(segment)
                shifted["start"] += start
                shifted["end"] += start
                merged.append(shifted)

        segments = _dedupe_segments(merged)
        if not segments:
            raise RuntimeError(
                "No speech segments returned by Gemini Transcribe"
            )

        registry = {}
        if settings.enable_video_analysis:
            report(
                "video",
                "Using bounded Gemini video understanding for character/acting analysis",
            )
            try:
                annotations, registry = VideoAnalyzer(
                    settings,
                    log,
                ).analyze(source_video, segments)
                _merge_video_annotations(
                    segments,
                    annotations,
                    registry,
                )
            except Exception as exc:
                log.warning(
                    "Video analysis failed; using audio diarization fallback: %s",
                    str(exc)[:1200],
                )
                registry = _build_fallback_registry(segments)
                _merge_video_annotations(segments, [], registry)
        else:
            report(
                "video",
                "Video analysis disabled; using audio diarization fallback",
            )
            registry = _build_fallback_registry(segments)
            _merge_video_annotations(segments, [], registry)

        fallback_registry = _build_fallback_registry(segments)
        for cid, fallback in fallback_registry.items():
            registry.setdefault(cid, fallback)

        # Rank characters by actual screen/dialogue presence so main characters
        # receive distinct voices before minor/background speakers.
        stats = {}
        for segment in segments:
            cid = segment["character_id"]
            stats.setdefault(cid, {"count": 0, "duration": 0.0})
            stats[cid]["count"] += 1
            stats[cid]["duration"] += max(
                0.0,
                float(segment["end"]) - float(segment["start"]),
            )
        for cid, profile in registry.items():
            score = stats.get(cid, {"count": 0, "duration": 0.0})
            profile["priority"] = (
                score["count"] * 2.0 + score["duration"]
            )

        voice_map = gemini.choose_voices(registry)
        for segment in segments:
            segment["voice"] = voice_map[segment["character_id"]]

        state["characters"] = {
            cid: {
                **registry.get(cid, {}),
                "voice": voice_map[cid],
            }
            for cid in voice_map
        }
        report(
            "video",
            f"Locked {len(voice_map)} character voices and acting profiles",
        )

        report(
            "translation",
            "Generating duration-aware Hindi dialogue",
        )
        for batch_no, batch in enumerate(
            _batch_by_chars(
                segments,
                settings.translation_batch_chars,
            ),
            1,
        ):
            report(
                "translation",
                f"Translation batch {batch_no}",
            )
            translated = gemini.translate_for_duration(batch)
            for segment in batch:
                segment["hindi"] = translated[segment["id"]]

        audio_items = []
        ordered_segments = sorted(
            segments,
            key=lambda x: float(x["start"]),
        )
        for i, segment in enumerate(ordered_segments):
            segment["next_start"] = (
                float(ordered_segments[i + 1]["start"])
                if i + 1 < len(ordered_segments)
                else duration
            )

        report(
            "tts",
            "Synthesizing each dialogue line with locked character voice and acting style",
        )
        for idx, segment in enumerate(segments, 1):
            report(
                "tts",
                f"Dialogue {idx}/{len(segments)}",
            )
            audio_path = _generate_segment_audio(
                gemini,
                segment,
                voice_map[segment["character_id"]],
                segments_dir,
                log,
            )
            segment["audio_path"] = str(audio_path)
            audio_items.append(
                (segment["start"], segment["dub_end"], audio_path)
            )

        report(
            "mix",
            "Building Hindi dialogue timeline",
        )
        dialogue_track = audio_dir / "hindi_dialogue.wav"
        assemble_track(
            audio_items,
            duration,
            dialogue_track,
        )

        report(
            "background",
            "Reconstructing background music/ambience from dialogue-free cues",
        )
        background = reconstruct_background(
            mix_audio,
            source_video,
            segments,
            work,
            log,
        )

        report(
            "render",
            "Rendering final movie with background/SFX preserved when available",
        )
        output_video.parent.mkdir(parents=True, exist_ok=True)
        mix_final(
            source_video,
            dialogue_track,
            background,
            output_video,
            segments=segments,
            log=log,
        )

        state["segments"] = segments
        state["status"] = "completed"
        state["stage"] = "done"
        _state(manifest, state)
        report("done", "High-quality Hindi dub is ready")
        return output_video

    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        _state(manifest, state)
        log.exception("Pipeline failed")
        if progress:
            try:
                progress("error", str(exc))
            except Exception:
                pass
        raise
