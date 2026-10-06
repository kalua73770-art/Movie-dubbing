from __future__ import annotations

import json
import logging
import re
import uuid
import threading
import time
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
    silence_ranges,
    ffprobe_duration,
    probe_wav,
    run_cmd,
    split_audio,
)
from hindi_dubbing.core.mixing.mixer import mix_final
from hindi_dubbing.openrouter import OpenRouterService
from hindi_dubbing.gemini import GeminiService
from hindi_dubbing.groq_translation import GroqTranslator
from hindi_dubbing.movie_brain import MovieBrain
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


def _generate_segment_audio(
    gemini,
    segment,
    voice,
    segment_dir,
    log,
    preferred_key_index: int | None = None,
    preferred_model_index: int | None = None,
    context_text: str = "",
    previous_interaction_id: str | None = None,
):
    preferred = max(0.25, float(segment["end"]) - float(segment["start"]))
    next_start = float(segment.get("next_start", segment["start"] + preferred))
    available = max(preferred, next_start - float(segment["start"]) - 0.06)

    natural_min = max(0.30, preferred * 0.82)
    natural_cap = min(available, max(preferred * 1.18, preferred + 1.0))
    # Allow the configured duration-rewrite budget; keep a bounded retry count.
    rewrite_attempts = max(0, min(int(getattr(settings, "rewrite_attempts", 1)), 3))

    best_path = None
    best_distance = float("inf")
    best_duration = 0.0

    for attempt in range(rewrite_attempts + 1):
        raw = segment_dir / f'{segment["id"]}_attempt{attempt}.wav'
        attempt_model_index = (
            (preferred_model_index + attempt) % len(settings.tts_models)
            if preferred_model_index is not None and settings.tts_models
            else preferred_model_index
        )
        attempt_key_index = (
            (preferred_key_index + attempt) % len(settings.api_keys)
            if preferred_key_index is not None and settings.api_keys
            else preferred_key_index
        )
        gemini.tts_segment(
            segment,
            voice,
            raw,
            preferred_key_index=attempt_key_index,
            preferred_model_index=attempt_model_index,
            context_text=context_text,
            previous_interaction_id=previous_interaction_id,
        )

        trimmed = segment_dir / f'{segment["id"]}_attempt{attempt}_trim.wav'
        trim_edge_silence(raw, trimmed, log)
        observed = probe_wav(trimmed)[3]

        distance = (
            natural_min - observed
            if observed < natural_min
            else observed - natural_cap
            if observed > natural_cap
            else 0.0
        )
        if distance < best_distance:
            best_path = trimmed
            best_distance = distance
            best_duration = observed

        log.info(
            "Dialogue %s preferred=%.3fs min=%.3fs cap=%.3fs available=%.3fs "
            "generated=%.3fs attempt=%d",
            segment["id"], preferred, natural_min, natural_cap,
            available, observed, attempt,
        )

        if natural_min <= observed <= natural_cap:
            segment["tts_duration"] = observed
            segment["dub_end"] = min(float(segment["start"]) + observed, next_start - 0.02)
            segment["tts_time_stretch"] = 1.0
            return trimmed

        if attempt < rewrite_attempts:
            segment["hindi"] = gemini.rewrite_for_observed_duration(
                segment,
                observed,
                target_duration=(natural_min if observed < natural_min else natural_cap),
            )

    if best_path is None:
        raise RuntimeError(f"TTS returned no audio for {segment['id']}")

    # Never discard a successfully generated line. A short line leaves the
    # reconstructed background running underneath; a long line may overlap the
    # next speaker naturally rather than being cut off.
    if best_duration < natural_min:
        segment["tts_duration"] = best_duration
        segment["dub_end"] = min(float(segment["start"]) + best_duration, next_start - 0.02)
        segment["tts_time_stretch"] = 1.0
        segment["duration_gap"] = max(0.0, preferred - best_duration)
        log.warning(
            "Dialogue %s shorter than target: %.3fs vs %.3fs; keeping full natural TTS",
            segment["id"], best_duration, preferred,
        )
        return best_path

    if best_duration > natural_cap:
        ratio = best_duration / natural_cap if natural_cap else 99.0
        if 0.65 <= ratio <= 1.55:
            fit = segment_dir / f'{segment["id"]}.wav'
            fit_audio(best_path, fit, natural_cap, log)
            fitted = probe_wav(fit)[3]
            segment["tts_duration"] = fitted
            segment["dub_end"] = float(segment["start"]) + fitted
            segment["tts_time_stretch"] = ratio
            return fit

        segment["tts_duration"] = best_duration
        segment["dub_end"] = float(segment["start"]) + best_duration
        segment["tts_time_stretch"] = 1.0
        segment["duration_overlap"] = max(0.0, best_duration - available)
        log.warning(
            "Dialogue %s longer than slot: %.3fs vs %.3fs; keeping full speech",
            segment["id"], best_duration, available,
        )
        return best_path


def _tts_batches(segments, max_chars: int, max_segments: int):
    ordered = sorted(segments, key=lambda x: float(x["start"]))
    # Single-turn mode is the production-safe default. One line per request avoids
    # ambiguous silence-boundary splitting that previously clipped long dialogue.
    if settings.tts_mode == "single":
        return [[seg] for seg in ordered]

    batches = []
    current = []
    chars = 0
    speakers = set()
    for seg in ordered:
        cost = len(seg.get("hindi", "")) + 60
        seg_speaker = str(seg.get("speaker", ""))
        if current and (
            chars + cost > max_chars
            or len(current) >= max_segments
            or (seg_speaker not in speakers and len(speakers) >= 2)
        ):
            batches.append(current)
            current = []
            chars = 0
            speakers = set()
        current.append(seg)
        chars += cost
        speakers.add(seg_speaker)
    if current:
        batches.append(current)
    return batches


def _split_batch_audio(
    batch_audio: Path,
    batch: list[dict],
    segment_dir: Path,
    log,
) -> list[Path] | None:
    # The TTS prompt inserts short pauses between turns. We use those pauses to
    # recover per-line clips; if the model does not preserve separations cleanly,
    # return None and let the caller fall back to safe per-line generation.
    ranges = silence_ranges(
        batch_audio,
        log=log,
        noise="-38dB",
        min_silence=0.22,
    )
    if len(ranges) < len(batch):
        return None
    if len(ranges) > len(batch):
        # Merge the shortest extra ranges into their nearest neighbour.
        ranges = list(ranges[:len(batch)])
    out = []
    for seg, (start, end) in zip(batch, ranges):
        path = segment_dir / f'{seg["id"]}_batch.wav'
        extract_range(batch_audio, path, start, end, log)
        trim_edge_silence(path, path.with_name(path.stem + "_trim.wav"), log)
        path = path.with_name(path.stem + "_trim.wav")
        observed = probe_wav(path)[3]
        target = max(0.25, float(seg["end"]) - float(seg["start"]))
        coverage = observed / target if target else 1.0
        if settings.tts_mode == "single" and coverage < settings.tts_min_coverage_ratio:
            log.warning(
                "Single-turn TTS clip too short for %s: %.3fs/%.3fs (%.1f%%); rejecting clipped audio",
                seg["id"], observed, target, coverage * 100.0,
            )
            return None
        seg["tts_duration"] = observed
        seg["dub_end"] = min(
            float(seg["start"]) + observed,
            float(seg.get("next_start", seg["start"] + observed)) - 0.02,
        )
        seg["tts_time_stretch"] = 1.0
        out.append(path)
    return out


def _generate_tts_batch_once(
    gemini,
    batch,
    voice_map,
    segment_dir,
    log,
    batch_index,
    model,
    key_index,
):
    out = segment_dir / f"tts_batch_{batch_index:04d}_m{model.replace('.', '_')}_k{key_index+1}.wav"
    speakers = {str(s.get("speaker", "")) for s in batch}
    if len(speakers) > 2:
        raise ValueError("TTS batch has more than two speakers")
    voices = {
        str(s.get("speaker", "")): voice_map[s["character_id"]]
        for s in batch
    }
    context = "\n".join(
        gemini._style(s)
        for s in batch[:4]
    )
    gemini.tts_batch_on_lane(
        batch,
        voices,
        out,
        model=model,
        key_index=key_index,
        context_text=context,
    )
    return _split_batch_audio(out, batch, segment_dir, log)


def _generate_tts_batch(
    gemini,
    batch,
    voice_map,
    segment_dir,
    log,
    batch_index,
    lanes,
):
    """Try a batch on concrete model/key lanes without serially probing all lanes."""
    last = None
    tried = set()
    max_attempts = min(4, max(1, len(lanes)))
    offset = batch_index % max(1, len(lanes))

    for attempt_no in range(max_attempts):
        lane = lanes[(offset + attempt_no) % len(lanes)]
        if lane in tried:
            continue
        tried.add(lane)
        model, key_index = lane
        try:
            paths = _generate_tts_batch_once(
                gemini,
                batch,
                voice_map,
                segment_dir,
                log,
                batch_index,
                model,
                key_index,
            )
            if paths is None:
                raise RuntimeError("TTS batch audio did not contain enough silence boundaries")
            return paths, lane
        except Exception as exc:
            last = exc
            log.warning(
                "TTS batch %d lane failed model=%s key=#%d: %s",
                batch_index + 1, model, key_index + 1, str(exc)[:800],
            )
    raise RuntimeError(
        f"TTS batch {batch_index + 1} failed after {len(tried)} concrete lane attempt(s): {last}"
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
    brain = MovieBrain(work / "movie_brain.json").load()
    state = {
        "job_id": job_id,
        "status": "running",
        "stage": "starting",
        "source": str(source_video),
        "output": str(output_video),
        "segments": [],
        "characters": {},
        "models": {},
        "brain": brain.snapshot(),
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
        groq = GroqTranslator(settings, log)

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
        reasoning_ok = False
        annotations = []

        if settings.enable_openrouter_continuity:
            report("brain", "Nemotron Nano Omni observer + Nemotron Ultra continuity brain")
            try:
                openrouter = OpenRouterService(settings, log)
                boundary_planner = VideoAnalyzer(settings, log)
                boundaries = boundary_planner.suggest_natural_boundaries(
                    source_video, segments, duration,
                    target_seconds=settings.openrouter_chunk_seconds,
                    max_seconds=settings.openrouter_chunk_max_seconds,
                    search_seconds=settings.openrouter_boundary_search_seconds,
                    work_dir=work,
                )
                video_chunks = openrouter.create_video_chunks(
                    source_video, boundaries, work / "openrouter_chunks"
                )
                report("brain", f"Prepared {len(video_chunks)} contiguous natural chunks")

                for chunk_index, chunk_info in enumerate(video_chunks, 1):
                    chunk_start=float(chunk_info["start"])
                    chunk_end=float(chunk_info["end"])
                    chunk_segments=[s for s in segments if float(s["start"])>=chunk_start-0.01 and float(s["end"])<=chunk_end+0.01]
                    if not chunk_segments:
                        continue
                    try:
                        observed=openrouter.analyze_chunk(
                            chunk_info["video"], chunk_info["audio"], chunk_segments,
                            brain.agent_context(max_chars=90000), chunk_start, chunk_end
                        )
                    except Exception as nano_exc:
                        log.warning("Nano Omni chunk %d/%d failed; Gemini fallback: %s",chunk_index,len(video_chunks),str(nano_exc)[:1000])
                        local=[]
                        for s in chunk_segments:
                            x=dict(s); x["start"]=max(0.0,float(s["start"])-chunk_start); x["end"]=max(0.0,float(s["end"])-chunk_start); local.append(x)
                        observed=VideoAnalyzer(settings,log).analyze(chunk_info["video"],local)
                    try:
                        continuity_update=openrouter.reason_continuity(
                            observed, chunk_segments, brain.agent_context(max_chars=100000),
                            chunk_start, chunk_end
                        )
                    except Exception as ultra_exc:
                        log.warning("Nemotron Ultra chunk %d/%d failed; observer output only: %s",chunk_index,len(video_chunks),str(ultra_exc)[:1000])
                        continuity_update=observed

                    valid_ids={str(s["id"]) for s in chunk_segments}
                    accepted=brain.apply_agent_update(continuity_update,segment_ids=valid_ids)
                    for ann in accepted:
                        sid=str(ann.get("segment_id"))
                        target=next((s for s in chunk_segments if str(s["id"])==sid),None)
                        if target:
                            for key in ("character_id","emotion","pace","intensity","style","on_screen"):
                                if key in ann and ann[key] not in (None,""):
                                    target[key]=ann[key]

                    registry={cid:dict(profile) for cid,profile in brain.data.get("characters",{}).items()}
                    brain.lock_voices(gemini.choose_voices(registry,locked_voices=brain.data.get("voice_locks",{})))
                    brain.save()
                    annotations.extend(accepted)
                    reasoning_ok=reasoning_ok or bool(accepted or registry)
                    report("brain",f"Chunk {chunk_index}/{len(video_chunks)} complete; characters={len(registry)} locked_voices={len(brain.data.get('voice_locks',{}))}")
                _merge_video_annotations(segments,annotations,registry)
            except Exception as exc:
                log.warning("OpenRouter continuity unavailable; Gemini fallback: %s",str(exc)[:1200])

        if not reasoning_ok and settings.enable_video_analysis:
            report("video","Using bounded Gemini video understanding as continuity fallback")
            try:
                video_result=VideoAnalyzer(settings,log).analyze(source_video,segments)
                annotations=video_result["annotations"]
                registry=video_result["registry"]
                for scene in video_result.get("scene_summaries",[]):
                    brain.add_scene_summary(scene.get("start",0),scene.get("end",0),scene.get("summary",""))
                brain.add_relationships(video_result.get("relationships",[]))
                brain.add_glossary(video_result.get("glossary",{}))
                _merge_video_annotations(segments,annotations,registry)
            except Exception as exc:
                log.warning("Gemini video analysis failed; audio diarization fallback: %s",str(exc)[:1200])
                registry=_build_fallback_registry(segments)
                _merge_video_annotations(segments,[],registry)
        elif not reasoning_ok:
            report("video","Scene reasoning disabled; audio diarization fallback")
            registry=_build_fallback_registry(segments)
            _merge_video_annotations(segments,[],registry)

        fallback_registry = _build_fallback_registry(segments)
        for cid, fallback in fallback_registry.items():
            registry.setdefault(cid, fallback)
        brain.merge_characters(registry)

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

        voice_map = gemini.choose_voices(
            registry,
            locked_voices=brain.data.get("voice_locks", {}),
        )
        voice_map = brain.lock_voices(voice_map)
        for segment in segments:
            segment["voice"] = voice_map[segment["character_id"]]
        brain.add_decision("Character voice assignments stay locked once established.")
        brain.save()

        state["characters"] = {
            cid: {
                **registry.get(cid, {}),
                "voice": voice_map[cid],
            }
            for cid in voice_map
        }
        state["brain"] = brain.snapshot()
        brain.save()
        report(
            "brain",
            f"Locked {len(voice_map)} character voices and acting profiles",
        )

        report(
            "translation",
            "Generating duration-aware Hindi dialogue",
        )
        translation_batches = _batch_by_chars(
            segments,
            settings.translation_batch_chars,
        )
        translation_workers = min(
            max(1, int(settings.translation_workers)),
            max(1, len(settings.api_keys)),
            max(1, len(translation_batches)),
        )
        report(
            "translation",
            f"Translating {len(segments)} lines in {len(translation_batches)} "
            f"large batch(es) with {translation_workers} worker(s)",
        )

        def translate_one(batch_index, batch):
            translated = groq.translate_for_duration(
                batch,
                preferred_key_index=batch_index % max(1, len(settings.api_keys)),
                preferred_model_index=batch_index,
                context_text=brain.context_for_segments(batch),
                previous_interaction_id=None,
            )
            return batch_index, translated

        with ThreadPoolExecutor(max_workers=translation_workers) as pool:
            translation_futures = [
                pool.submit(translate_one, batch_index, batch)
                for batch_index, batch in enumerate(translation_batches)
            ]
            translated_done = 0
            for future in as_completed(translation_futures):
                batch_index, translated = future.result()
                for segment in translation_batches[batch_index]:
                    segment["hindi"] = translated[segment["id"]]
                translated_done += 1
                report(
                    "translation",
                    f"Completed translation batch {translated_done}/{len(translation_batches)}",
                )

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

        if settings.tts_mode == "single":
            ordered_tts = sorted(segments, key=lambda x: float(x["start"]))
            tts_workers = min(
                max(1, int(settings.tts_workers)),
                max(1, len(ordered_tts)),
            )
            report(
                "tts",
                f"Synthesizing {len(ordered_tts)} dialogue lines individually "
                f"with {tts_workers} workers",
            )

            def synthesize_line(idx, segment):
                preferred_model_index = idx % max(1, len(settings.tts_models))
                preferred_key_index = idx % max(1, len(settings.api_keys))
                voice = voice_map[segment["character_id"]]
                audio_path = _generate_segment_audio(
                    gemini,
                    segment,
                    voice,
                    segments_dir,
                    log,
                    preferred_key_index=preferred_key_index,
                    preferred_model_index=preferred_model_index,
                    context_text=brain.character_context(segment["character_id"]),
                    previous_interaction_id=None,
                )
                target = max(0.25, float(segment["end"]) - float(segment["start"]))
                observed = float(segment.get("tts_duration", 0.0) or 0.0)
                coverage = observed / target if target else 1.0
                if coverage < settings.tts_min_coverage_ratio:
                    # Duration QA is a warning, not a reason to discard the whole movie.
                    # _generate_segment_audio already performs bounded rewrite retries.
                    segment["tts_quality_warning"] = {
                        "coverage": coverage,
                        "target": target,
                        "observed": observed,
                    }
                    log.warning(
                        "TTS coverage below target for %s: %.3fs/%.3fs (%.1f%%); "
                        "keeping best natural TTS after bounded retries",
                        segment["id"], observed, target, coverage * 100.0,
                    )
                return segment, audio_path

            tts_results_by_id = {}
            with ThreadPoolExecutor(max_workers=tts_workers) as pool:
                futures = {
                    pool.submit(synthesize_line, idx, segment): idx
                    for idx, segment in enumerate(ordered_tts)
                }
                completed = 0
                for future in as_completed(futures):
                    segment, audio_path = future.result()
                    segment["audio_path"] = str(audio_path)
                    tts_results_by_id[segment["id"]] = audio_path
                    completed += 1
                    report(
                        "tts",
                        f"Completed dialogue {completed}/{len(ordered_tts)} "
                        f"({segment['id']})",
                    )
            tts_results = [tts_results_by_id[segment["id"]] for segment in segments]

        else:
            tts_batches = _tts_batches(
                segments,
                int(settings.tts_batch_chars),
                int(settings.tts_batch_max_segments),
            )
            lanes = gemini.tts_lanes()
            if not lanes:
                raise RuntimeError("No TTS model/key lanes survived preflight")
            tts_workers = min(
                max(1, int(settings.tts_workers)),
                max(1, len(tts_batches)),
                max(1, len(lanes)),
            )
            report(
                "tts",
                f"Synthesizing {len(segments)} lines as {len(tts_batches)} batches "
                f"across {len(lanes)} model/key lanes with {tts_workers} workers",
            )

            lane_lock = {}
            lane_last_request = {}
            lane_guard = threading.Lock()

            def synthesize_batch(idx, batch):
                last = None
                order = [lanes[(idx + shift) % len(lanes)] for shift in range(len(lanes))]
                for lane in order:
                    model, key_index = lane
                    try:
                        with lane_guard:
                            while lane_lock.get(lane, False):
                                lane_guard.release()
                                time.sleep(0.05)
                                lane_guard.acquire()
                            lane_lock[lane] = True
                            last_at = lane_last_request.get(lane, 0.0)
                            wait = max(
                                0.0,
                                float(settings.tts_lane_cooldown_seconds)
                                - (time.monotonic() - last_at),
                            )
                        if wait:
                            time.sleep(wait)

                        try:
                            paths, used_lane = _generate_tts_batch(
                                gemini,
                                batch,
                                voice_map,
                                segments_dir,
                                log,
                                idx,
                                [lane],
                            )
                        finally:
                            with lane_guard:
                                lane_last_request[lane] = time.monotonic()
                                lane_lock[lane] = False
                        return idx, list(zip(batch, paths)), used_lane
                    except Exception as exc:
                        last = exc
                        log.warning(
                            "TTS batch %d moving to next lane after model=%s key=#%d: %s",
                            idx + 1,
                            model,
                            key_index + 1,
                            str(exc)[:700],
                        )
                        continue

                raise RuntimeError(
                    f"TTS batch {idx + 1} failed on all available lanes: {last}"
                )

            tts_results = [None] * len(segments)
            with ThreadPoolExecutor(max_workers=tts_workers) as pool:
                futures = {
                    pool.submit(synthesize_batch, idx, batch): idx
                    for idx, batch in enumerate(tts_batches)
                }
                done = 0
                for future in as_completed(futures):
                    idx, items, lane = future.result()
                    for segment, audio_path in items:
                        segment["audio_path"] = str(audio_path)
                        tts_results[segments.index(segment)] = audio_path
                    done += 1
                    report(
                        "tts",
                        f"Completed TTS batch {done}/{len(tts_batches)} on "
                        f"{lane[0]} key=#{lane[1]+1}",
                    )

        for segment, audio_path in zip(segments, tts_results):
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
        state["brain"] = brain.snapshot()
        brain.save()
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