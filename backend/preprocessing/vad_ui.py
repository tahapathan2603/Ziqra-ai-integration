"""
Gradio development UI for manually testing the Silero VAD + speech-to-text pipeline.

This module holds no VAD/STT logic of its own — it receives the pipeline function(s)
from silero_vad.py via launch() and only handles presentation (audio input,
triggering the pipeline, and displaying segments/transcripts/chunks/logs). This is
dev/testing tooling only; it will be removed or replaced once the pipeline is wired
into the FastAPI backend.
"""

import glob
import json
import logging
import os
import time
from collections import Counter

import gradio as gr

# Gradio can't render a variable-length list of output components, so we
# pre-allocate this many audio player slots and toggle visibility per run.
MAX_DISPLAYED_CHUNKS = 25


def _format_timestamp(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_timestamp_fractional(seconds: float) -> str:
    """Format seconds as MM:SS.ff (fractional seconds), for word-level timing."""
    total_seconds = max(0.0, seconds)
    minutes = int(total_seconds // 60)
    secs = total_seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def _clear_existing_chunks(chunks_dir: str) -> None:
    for path in glob.glob(os.path.join(chunks_dir, "chunk_*.wav")):
        os.remove(path)


def _format_sentences(sentences):
    if not sentences:
        return "No sentences detected."
    return "\n\n".join(
        f"Sentence {i}\n"
        f"Start: {_format_timestamp(s['start'])}\n"
        f"End: {_format_timestamp(s['end'])}\n"
        f"Text:\n{s['text']}"
        for i, s in enumerate(sentences, start=1)
    )


def _format_words(words):
    if not words:
        return "No words detected."
    max_word_len = max(len(w["word"]) for w in words)
    return "\n".join(
        f"{w['word']:<{max_word_len}s} → "
        f"{_format_timestamp_fractional(w['start'])} - {_format_timestamp_fractional(w['end'])}"
        for w in words
    )


def _format_classification(value: str) -> str:
    return value.replace("_", " ").title()


def _format_fillers(fillers_report):
    fillers = fillers_report["fillers"]
    if fillers:
        most_common_word, _ = Counter(f["word"] for f in fillers).most_common(1)[0]
        most_common_line = f'Most common filler: "{most_common_word}"'
    else:
        most_common_line = "Most common filler: none"
    return (
        f"Fillers detected: {fillers_report['filler_count']}\n"
        f"{most_common_line}\n"
        f"Classification: {_format_classification(fillers_report['classification'])}"
    )


def _format_pauses(pauses_report):
    pauses = pauses_report["pauses"]
    long_pauses = sum(1 for p in pauses if p["type"] == "long_pause")
    dead_air = sum(1 for p in pauses if p["type"] == "dead_air")
    return f"Long pauses: {long_pauses}\nDead air instances: {dead_air}"


def _format_speaking_speed(speed_report):
    return (
        f"Words per minute: {speed_report['words_per_minute']}\n"
        f"Classification: {_format_classification(speed_report['classification'])}"
    )


def _format_pronunciation_summary(pronunciation_report):
    r = pronunciation_report
    scores = (
        f"Pronunciation Score: {r['pronunciation_score']}  (phoneme accuracy + rhythm)\n"
        f"Phoneme Accuracy: {round(r['phoneme_accuracy'] * 100)}%\n"
        f"Stress Accuracy: {round(r['stress_accuracy'] * 100)}%  "
        f"[PLACEHOLDER detection — excluded from the score above]\n"
        f"Rhythm Score: {round(r['rhythm_score'] * 100)}%"
    )
    return f"{scores}\n\nOverall:\n{_format_observations(r['overall_observations'])}"


def _format_phoneme_analysis(pronunciation_report):
    accuracy_line = f"Phoneme Accuracy: {round(pronunciation_report['phoneme_accuracy'] * 100)}%"
    errors = pronunciation_report["phoneme_errors"]
    if not errors:
        return f"{accuracy_line}\n\nNo phoneme errors detected."
    error_blocks = "\n\n".join(
        f"Word: {e['word']}\n"
        f"Expected phoneme: {e['expected'] or '(none — extra phoneme detected)'}\n"
        f"Detected phoneme: {e['detected'] or '(none — phoneme missing)'}"
        for e in errors
    )
    return f"{accuracy_line}\n\n{error_blocks}"


def _format_mispronounced_words(pronunciation_report):
    mispronounced = pronunciation_report["mispronounced_words"]
    if not mispronounced:
        return "No mispronounced words detected."
    max_len = max(len(w["word"]) for w in mispronounced)
    lines = []
    for w in mispronounced:
        start = f"{w['start']:.2f}s" if w["start"] is not None else "?"
        end = f"{w['end']:.2f}s" if w["end"] is not None else "?"
        lines.append(
            f"{w['word']:<{max_len}s} → {w['severity']} severity  (Start: {start}, End: {end})"
        )
    return "\n".join(lines)


def _format_stress_analysis(pronunciation_report):
    accuracy_line = (
        f"Stress Accuracy: {round(pronunciation_report['stress_accuracy'] * 100)}%  "
        f"[PLACEHOLDER detection — see stress_placement.py]"
    )
    errors = pronunciation_report["stress_errors"]
    if not errors:
        return f"{accuracy_line}\n\nNo stress errors detected."
    error_blocks = "\n\n".join(
        f"Word: {e['word']}\nExpected: {e['expected']}\nDetected: {e['detected']}" for e in errors
    )
    return f"{accuracy_line}\n\n{error_blocks}"


def _format_rhythm_analysis(pronunciation_report):
    score_line = f"Rhythm Score: {round(pronunciation_report['rhythm_score'] * 100)}%"
    issues = pronunciation_report["rhythm_issues"]
    if not issues:
        return f"{score_line}\n\nIssues: none"
    issues_block = "\n".join(f"- {issue}" for issue in issues)
    return f"{score_line}\n\nIssues:\n{issues_block}"


def _format_raw_pronunciation_json(pronunciation_report):
    return json.dumps(pronunciation_report, indent=2, ensure_ascii=False)


def _format_mti_summary(mti_report):
    # .get(), not bracket access: Fix 7 (audio-pipeline correctness-fix plan)
    # omits "top_recurring_issue" entirely when there's no recurring pattern,
    # rather than carrying a None — direct indexing would KeyError on that.
    top_recurring_issue = mti_report.get("top_recurring_issue")
    scores = (
        f"MTI Score: {mti_report['overall_mti_score']}\n"
        f"Clarity Impact: {_format_classification(mti_report['clarity_impact'])}\n"
        f"Top Recurring Issue: "
        f"{_format_classification(top_recurring_issue) if top_recurring_issue else 'None'}"
    )
    return f"{scores}\n\nOverall:\n{_format_observations(mti_report['overall_observations'])}"


def _format_mti_vowel_patterns(mti_report):
    patterns = mti_report["vowel_patterns"]
    if not patterns:
        return "No vowel pattern issues detected."
    return "\n\n".join(
        f"Word: {p['word']}\n"
        f"Issue: {_format_classification(p['issue'])}\n"
        f"Expected: {p['expected']}\n"
        f"Detected: {p['detected']}\n"
        f"Severity: {_format_classification(p['severity'])}"
        for p in patterns
    )


def _format_mti_consonant_patterns(mti_report):
    patterns = mti_report["consonant_patterns"]
    if not patterns:
        return "No consonant pattern issues detected."
    return "\n\n".join(
        f"Word: {p['word']}\n"
        f"Issue: {_format_classification(p['issue'])}\n"
        f"Expected: {p['expected']}\n"
        f"Detected: {p['detected']}\n"
        f"Severity: {_format_classification(p['severity'])}"
        for p in patterns
    )


def _format_mti_stress_transfer(mti_report):
    transfers = mti_report["stress_transfer"]
    if not transfers:
        return "No stress transfer issues detected.  [PLACEHOLDER-DEPENDENT — see stress_placement.py]"
    return "\n\n".join(
        f"Word: {t['word']}\n"
        f"Expected: {t['expected_stress']}\n"
        f"Detected: {t['detected_stress']}\n"
        f"Severity: {_format_classification(t['severity'])}"
        for t in transfers
    )


def _format_mti_speech_statistics(mti_report):
    stats = mti_report["speech_statistics"]
    return (
        f"Total Words Analyzed: {stats['total_words_analyzed']}\n"
        f"Affected Words: {stats['affected_words']}\n"
        f"Affected Percentage: {stats['affected_percentage']}%"
    )


def _format_mti_patterns_detected(mti_report):
    patterns = mti_report["patterns_detected"]
    if not patterns:
        return "No patterns detected."
    return "\n".join(f"- {p}" for p in patterns)


def _format_raw_mti_json(mti_report):
    return json.dumps(mti_report, indent=2, ensure_ascii=False)


def _format_observations(observations):
    if not observations:
        return "None."
    return "\n".join(f"• {o}" for o in observations)


def _format_intonation_summary(intonation_report):
    # Read the delivery label straight from the report (intonation_analyzer.py
    # owns the label vocabulary) rather than re-deriving it here — a second copy
    # of the thresholds drifted out of sync with the analyzer's own labels.
    score = intonation_report["intonation_score"]
    scores = f"Intonation Score: {score}\nOverall Delivery: {intonation_report['delivery_label']}"
    return f"{scores}\n\nOverall:\n{_format_observations(intonation_report['overall_observations'])}"


def _format_pitch_analysis(intonation_report):
    p = intonation_report["pitch_variation"]
    return (
        f"Pitch Score: {p['pitch_score']}\n"
        f"Average Pitch: {p['average_pitch']} Hz\n"
        f"Min Pitch: {p['min_pitch']} Hz\n"
        f"Max Pitch: {p['max_pitch']} Hz\n"
        f"Pitch Range: {p['pitch_range']} Hz\n\n"
        f"Observations:\n{_format_observations(p['observations'])}"
    )


def _format_energy_analysis(intonation_report):
    e = intonation_report["energy_variation"]
    sections = e["low_energy_sections"]
    if sections:
        sections_text = "\n".join(
            f"{s['start']}s → {s['end']}s ({_format_classification(s['severity'])} severity)" for s in sections
        )
    else:
        sections_text = "None."
    return (
        f"Energy Score: {e['energy_score']}\n\n"
        f"Low-energy section(s):\n{sections_text}\n\n"
        f"Observations:\n{_format_observations(e['observations'])}"
    )


def _format_monotonicity_analysis(intonation_report):
    m = intonation_report["monotonicity"]
    sections = m["monotone_sections"]
    if sections:
        sections_text = "\n".join(
            f"{s['start']}s → {s['end']}s ({_format_classification(s['severity'])} severity)" for s in sections
        )
    else:
        sections_text = "None."
    return (
        f"Monotonicity Score: {m['monotonicity_score']}\n\n"
        f"Monotone section(s):\n{sections_text}\n\n"
        f"Observations:\n{_format_observations(m['observations'])}"
    )


def _format_emphasis_analysis(intonation_report):
    em = intonation_report["emphasis"]
    words = em["under_emphasized_words"]
    if words:
        words_text = "\n".join(f"- {w['word']}" for w in words)
    else:
        words_text = "None."
    return (
        f"Emphasis Score: {em['emphasis_score']}\n\n"
        f"Under-emphasized words:\n{words_text}\n\n"
        f"Observations:\n{_format_observations(em['observations'])}"
    )


def _format_raw_intonation_json(intonation_report):
    return json.dumps(intonation_report, indent=2, ensure_ascii=False)


def _format_engagement_level(level: str) -> str:
    return level.replace("_", " ").title()


def _format_engagement_summary(engagement_report):
    return (
        f"Engagement Score: {engagement_report['engagement_score']}\n"
        f"Engagement Level: {_format_engagement_level(engagement_report['engagement_level'])}"
    )


def _format_engagement_strengths(engagement_report):
    strengths = engagement_report["strengths"]
    if not strengths:
        return "None identified."
    return "\n".join(f"✓ {s}." if not s.endswith(".") else f"✓ {s}" for s in strengths)


def _format_engagement_improvements(engagement_report):
    improvements = engagement_report["improvement_areas"]
    if not improvements:
        return "None identified."
    return "\n".join(f"• {i}." if not i.endswith(".") else f"• {i}" for i in improvements)


def _format_engagement_timeline(engagement_report):
    timeline = engagement_report["timeline"]
    if not timeline:
        return "No timeline available (insufficient timestamped evidence to estimate recording duration)."
    return "\n".join(
        f"{t['start']}s → {t['end']}s : {_format_engagement_level(t['engagement'])} Engagement" for t in timeline
    )


def _format_engagement_energy_patterns(engagement_report):
    # Synthesis only: the score and the engagement takeaway. The per-segment
    # timings already appear in the Intonation → Energy panel and in the Raw
    # Engagement JSON, so they aren't repeated here.
    e = engagement_report["energy_patterns"]
    return (
        f"Energy Engagement Score: {e['energy_engagement_score']}\n\n"
        f"{_format_observations(e['observations'])}"
    )


def _format_engagement_pause_patterns(engagement_report):
    # Synthesis only: score, the net-new disengaging-pause / hesitation-cluster
    # takeaways (in observations), and a compact cluster count. Individual pause
    # timings live in the Fluency → Pauses panel and the Raw Engagement JSON.
    p = engagement_report["pause_patterns"]
    cluster_count = len(p.get("hesitation_clusters", []))
    cluster_line = f"\n\nHesitation clusters: {cluster_count}" if cluster_count else ""
    return (
        f"Pause Engagement Score: {p['pause_engagement_score']}\n\n"
        f"{_format_observations(p['observations'])}"
        f"{cluster_line}"
    )


def _format_engagement_speaking_dynamics(engagement_report):
    d = engagement_report["speaking_dynamics"]
    strengths_text = "\n".join(f"- {s}" for s in d["strengths"]) if d["strengths"] else "None."
    issues_text = "\n".join(f"- {i}" for i in d["issues"]) if d["issues"] else "None."
    return (
        f"Dynamics Score: {d['dynamics_score']}\n\n"
        f"Strengths:\n{strengths_text}\n\n"
        f"Issues:\n{issues_text}\n\n"
        f"Observations:\n{_format_observations(d['observations'])}"
    )


def _format_raw_engagement_json(engagement_report):
    return json.dumps(engagement_report, indent=2, ensure_ascii=False)


def _format_dataset_status(dataset_result):
    t = dataset_result["timeline"]
    return (
        f"Session ID: {dataset_result['session_id']}\n"
        f"Timeline: {len(t['words'])} words, {len(t['sentences'])} sentences, "
        f"{len(t['detected_phonemes'])} detected phonemes, "
        f"{len(t['acoustic_contours']['pitch_hz'])} pitch points, "
        f"{len(t['acoustic_contours']['energy_rms'])} energy points\n"
        f"Saved to:\n  {dataset_result['timeline_path']}\n  {dataset_result['features_path']}"
    )


def _format_timeline_json(dataset_result):
    return json.dumps(dataset_result["timeline"], indent=2, ensure_ascii=False)


def _format_features_json(dataset_result):
    return json.dumps(dataset_result["features"], indent=2, ensure_ascii=False)


def _format_output_matrix_json(dataset_result):
    return json.dumps(
        {"timeline": dataset_result["timeline"], "features": dataset_result["features"]},
        indent=2,
        ensure_ascii=False,
    )


def _format_articulation_packet_json(coach_packets):
    return json.dumps(coach_packets["articulation"], indent=2, ensure_ascii=False)


def _format_delivery_packet_json(coach_packets):
    return json.dumps(coach_packets["delivery"], indent=2, ensure_ascii=False)


def _format_feedback_status(feedback):
    return f"⚠ {feedback['error']}" if "error" in feedback else "Feedback generated."


def _format_feedback_overall(feedback):
    if "error" in feedback:
        return "No feedback available — see status above."
    return feedback.get("overall_assessment", "")


def _format_feedback_strengths(feedback):
    strengths = feedback.get("key_strengths", [])
    if not strengths:
        return "None." if "error" in feedback else "None identified."
    return "\n".join(f"✓ {s}" for s in strengths)


def _format_feedback_improvements(feedback):
    improvements = feedback.get("priority_improvements", [])
    if not improvements:
        return "None." if "error" in feedback else "None identified."
    return "\n\n".join(
        f"Issue: {i.get('issue', '')}\n"
        f"Evidence: {i.get('evidence', '')}\n"
        f"Suggestion: {i.get('suggestion', '')}"
        for i in improvements
    )


def _format_feedback_practice_tips(feedback):
    tips = feedback.get("practice_tips", [])
    if not tips:
        return "None." if "error" in feedback else "None identified."
    return "\n".join(f"• {t}" for t in tips)


def _format_raw_feedback_json(evidence_packet, feedback):
    return json.dumps({"evidence_packet": evidence_packet, "feedback": feedback}, indent=2, ensure_ascii=False)


def run_pipeline(
    audio_path,
    process_audio,
    save_audio_fn,
    chunks_dir,
    process_chunks,
    stt_model_name,
    process_timestamps,
    analyze_audio,
    build_and_save_dataset,
    dataset_dir,
    build_coach_packets,
):
    """
    Run the VAD + STT + timestamps + audio-analysis (Fluency/Pronunciation/MTI/
    Intonation/Engagement, parallelized — see feature_extractors/audio/
    audio_analyzer.py) pipeline on audio_path, persist the two-level audio
    dataset (see distillation/dataset_builder.py), build the two per-coach
    training packets (see feedback/coach_packets.py), and format everything for
    display.

    Returns (summary_text, segments_text, transcript_text, sentences_text, words_text,
    fillers_text, pauses_text, speaking_speed_text, pronunciation_summary_text,
    phoneme_analysis_text, mispronounced_words_text, stress_analysis_text,
    rhythm_analysis_text, pronunciation_json_text, mti_summary_text,
    mti_vowel_patterns_text, mti_consonant_patterns_text,
    mti_stress_transfer_text, mti_speech_statistics_text, mti_patterns_detected_text,
    mti_json_text, intonation_summary_text, pitch_analysis_text, energy_analysis_text,
    monotonicity_analysis_text, emphasis_analysis_text, intonation_json_text,
    engagement_summary_text, engagement_strengths_text, engagement_improvements_text,
    engagement_timeline_text, engagement_energy_patterns_text, engagement_pause_patterns_text,
    engagement_speaking_dynamics_text, engagement_json_text,
    dataset_status_text, timeline_json_text, articulation_packet_json_text,
    delivery_packet_json_text, features_json_text, output_matrix_json_text,
    debug_text, logs_text, chunk_paths, reports).

    reports is a dict of the five raw analyzer outputs (fluency, pronunciation,
    mti, intonation, engagement) plus transcript/total_duration — passed through
    for the AI Feedback section to build an evidence packet from, without
    re-running the pipeline.
    Kept separate from launch() so it can be called/tested without Gradio running.
    """
    log_records = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            log_records.append(self.format(record))

    list_handler = _ListHandler()
    list_handler.setFormatter(logging.Formatter("%(message)s"))
    list_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.addHandler(list_handler)
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)

    empty = "No audio provided."
    # 43 text placeholders — generated, not hand-counted, specifically to
    # avoid the kind of off-by-one mismatch against the success-path return
    # below that has bitten this exact function before. Verified equal by an
    # automated test (see verification suite) rather than by eyeballing.
    _EMPTY_TEXT_OUTPUT_COUNT = 43
    try:
        if not audio_path:
            return (empty,) * _EMPTY_TEXT_OUTPUT_COUNT + ([], {})

        result = process_audio(audio_path)
        vad_segments = result["speech_segments"]
        chunks = result["chunks"]
        sample_rate = result["sample_rate"]
        waveform = result["waveform"]
        total_duration = waveform.shape[-1] / sample_rate

        logging.getLogger(__name__).info("Saving chunks...")
        os.makedirs(chunks_dir, exist_ok=True)
        _clear_existing_chunks(chunks_dir)

        chunk_paths = []
        chunk_records = []
        for i, (chunk, seg) in enumerate(zip(chunks, vad_segments), start=1):
            chunk_path = os.path.join(chunks_dir, f"chunk_{i}.wav")
            save_audio_fn(chunk_path, chunk, sampling_rate=sample_rate)
            chunk_paths.append(chunk_path)
            chunk_records.append(
                {"chunk_id": i, "path": chunk_path, "start": seg["start"], "end": seg["end"]}
            )

        stt_start = time.time()
        # word_timestamps=True computes word-level timing in this same pass, so
        # process_timestamps() below doesn't have to re-transcribe the audio a
        # second time just to get it.
        stt_result = process_chunks(chunk_records, word_timestamps=True)
        stt_elapsed = time.time() - stt_start

        ts_result = process_timestamps(stt_result, chunk_records, words=stt_result.get("words"))

        speech_duration = sum(seg["end"] - seg["start"] for seg in vad_segments)

        # Fluency/Pronunciation/MTI/Intonation/Engagement now run through one
        # orchestrator, parallelized where real dependencies allow, instead of
        # five sequential calls here — see feature_extractors/audio/audio_analyzer.py
        # for the dependency graph and why MTI/Engagement can't simply all run
        # alongside the rest.
        audio_analysis = analyze_audio(
            audio_path,
            ts_result["full_transcript"],
            ts_result["words"],
            ts_result["sentences"],
            speech_duration,
            total_duration=total_duration,
            speech_chunks=[
                {"chunk_id": r["chunk_id"], "start": r["start"], "end": r["end"]} for r in chunk_records
            ],
        )
        fluency_report = audio_analysis["fluency"]
        pronunciation_report = audio_analysis["pronunciation"]
        mti_report = audio_analysis["mti"]
        intonation_report = audio_analysis["intonation"]
        engagement_report = audio_analysis["engagement"]

        logging.getLogger(__name__).info("Building and saving audio dataset (Level 1 + Level 2)...")
        dataset_result = build_and_save_dataset(
            duration=total_duration,
            sample_rate=sample_rate,
            detected_language=ts_result.get("detected_language"),
            language_probability=ts_result.get("language_probability"),
            speech_chunks=[
                {"chunk_id": r["chunk_id"], "start": r["start"], "end": r["end"]} for r in chunk_records
            ],
            sentences=ts_result["sentences"],
            words=ts_result["words"],
            audio_analysis=audio_analysis,
            dataset_dir=dataset_dir,
        )

        logging.getLogger(__name__).info("Building per-coach training packets (Articulation, Delivery)...")
        coach_packets = build_coach_packets(audio_analysis, session_id=dataset_result["session_id"])
    finally:
        root_logger.removeHandler(list_handler)
        root_logger.setLevel(previous_level)

    summary_text = (
        f"Total duration: {_format_duration(total_duration)}\n"
        f"Speech duration: {_format_duration(speech_duration)}\n"
        f"Speech segments detected: {len(vad_segments)}"
    )

    transcript_segments = stt_result["segments"]
    if vad_segments:
        segments_text = "\n\n".join(
            f"Segment {i}\n"
            f"Start: {_format_timestamp(seg['start'])}\n"
            f"End: {_format_timestamp(seg['end'])}\n"
            f"Duration: {_format_duration(seg['end'] - seg['start'])}\n"
            f"Transcript: {transcript_segments[i - 1]['text'] or '(no speech recognized)'}"
            for i, seg in enumerate(vad_segments, start=1)
        )
    else:
        segments_text = "No speech segments detected."

    transcript_text = stt_result["full_transcript"] or "(no speech recognized)"
    sentences_text = _format_sentences(ts_result["sentences"])
    words_text = _format_words(ts_result["words"])

    fillers_text = _format_fillers(fluency_report["fillers"])
    pauses_text = _format_pauses(fluency_report["pauses"])
    speaking_speed_text = _format_speaking_speed(fluency_report["speaking_speed"])

    pronunciation_summary_text = _format_pronunciation_summary(pronunciation_report)
    phoneme_analysis_text = _format_phoneme_analysis(pronunciation_report)
    mispronounced_words_text = _format_mispronounced_words(pronunciation_report)
    stress_analysis_text = _format_stress_analysis(pronunciation_report)
    rhythm_analysis_text = _format_rhythm_analysis(pronunciation_report)
    pronunciation_json_text = _format_raw_pronunciation_json(pronunciation_report)

    mti_summary_text = _format_mti_summary(mti_report)
    mti_vowel_patterns_text = _format_mti_vowel_patterns(mti_report)
    mti_consonant_patterns_text = _format_mti_consonant_patterns(mti_report)
    mti_stress_transfer_text = _format_mti_stress_transfer(mti_report)
    mti_speech_statistics_text = _format_mti_speech_statistics(mti_report)
    mti_patterns_detected_text = _format_mti_patterns_detected(mti_report)
    mti_json_text = _format_raw_mti_json(mti_report)

    intonation_summary_text = _format_intonation_summary(intonation_report)
    pitch_analysis_text = _format_pitch_analysis(intonation_report)
    energy_analysis_text = _format_energy_analysis(intonation_report)
    monotonicity_analysis_text = _format_monotonicity_analysis(intonation_report)
    emphasis_analysis_text = _format_emphasis_analysis(intonation_report)
    intonation_json_text = _format_raw_intonation_json(intonation_report)

    engagement_summary_text = _format_engagement_summary(engagement_report)
    engagement_strengths_text = _format_engagement_strengths(engagement_report)
    engagement_improvements_text = _format_engagement_improvements(engagement_report)
    engagement_timeline_text = _format_engagement_timeline(engagement_report)
    engagement_energy_patterns_text = _format_engagement_energy_patterns(engagement_report)
    engagement_pause_patterns_text = _format_engagement_pause_patterns(engagement_report)
    engagement_speaking_dynamics_text = _format_engagement_speaking_dynamics(engagement_report)
    engagement_json_text = _format_raw_engagement_json(engagement_report)

    dataset_status_text = _format_dataset_status(dataset_result)
    timeline_json_text = _format_timeline_json(dataset_result)
    articulation_packet_json_text = _format_articulation_packet_json(coach_packets)
    delivery_packet_json_text = _format_delivery_packet_json(coach_packets)
    features_json_text = _format_features_json(dataset_result)
    output_matrix_json_text = _format_output_matrix_json(dataset_result)

    debug_text = (
        f"Speech chunks detected: {len(chunks)}\n"
        f"Model used: Whisper {stt_model_name}\n"
        f"Total transcription time: {stt_elapsed:.1f} seconds"
    )

    logs_text = "\n".join(log_records)

    reports = {
        "fluency": fluency_report,
        "pronunciation": pronunciation_report,
        "mti": mti_report,
        "intonation": intonation_report,
        "engagement": engagement_report,
        "transcript": ts_result["full_transcript"],
        "total_duration": total_duration,
    }

    return (
        summary_text,
        segments_text,
        transcript_text,
        sentences_text,
        words_text,
        fillers_text,
        pauses_text,
        speaking_speed_text,
        pronunciation_summary_text,
        phoneme_analysis_text,
        mispronounced_words_text,
        stress_analysis_text,
        rhythm_analysis_text,
        pronunciation_json_text,
        mti_summary_text,
        mti_vowel_patterns_text,
        mti_consonant_patterns_text,
        mti_stress_transfer_text,
        mti_speech_statistics_text,
        mti_patterns_detected_text,
        mti_json_text,
        intonation_summary_text,
        pitch_analysis_text,
        energy_analysis_text,
        monotonicity_analysis_text,
        emphasis_analysis_text,
        intonation_json_text,
        engagement_summary_text,
        engagement_strengths_text,
        engagement_improvements_text,
        engagement_timeline_text,
        engagement_energy_patterns_text,
        engagement_pause_patterns_text,
        engagement_speaking_dynamics_text,
        engagement_json_text,
        dataset_status_text,
        timeline_json_text,
        articulation_packet_json_text,
        delivery_packet_json_text,
        features_json_text,
        output_matrix_json_text,
        debug_text,
        logs_text,
        chunk_paths,
        reports,
    )


def launch(
    process_audio,
    save_audio_fn,
    chunks_dir,
    process_chunks,
    stt_model_name,
    process_timestamps,
    analyze_audio,
    build_and_save_dataset,
    dataset_dir,
    build_coach_packets,
    build_evidence_packet,
    generate_feedback,
    host="127.0.0.1",
    port=7860,
):
    """Build and launch the Gradio VAD + STT + timestamps + fluency + pronunciation + MTI testing interface."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def on_run_clicked(audio_path):
        (
            summary_text,
            segments_text,
            transcript_text,
            sentences_text,
            words_text,
            fillers_text,
            pauses_text,
            speaking_speed_text,
            pronunciation_summary_text,
            phoneme_analysis_text,
            mispronounced_words_text,
            stress_analysis_text,
            rhythm_analysis_text,
            pronunciation_json_text,
            mti_summary_text,
            mti_vowel_patterns_text,
            mti_consonant_patterns_text,
            mti_stress_transfer_text,
            mti_speech_statistics_text,
            mti_patterns_detected_text,
            mti_json_text,
            intonation_summary_text,
            pitch_analysis_text,
            energy_analysis_text,
            monotonicity_analysis_text,
            emphasis_analysis_text,
            intonation_json_text,
            engagement_summary_text,
            engagement_strengths_text,
            engagement_improvements_text,
            engagement_timeline_text,
            engagement_energy_patterns_text,
            engagement_pause_patterns_text,
            engagement_speaking_dynamics_text,
            engagement_json_text,
            dataset_status_text,
            timeline_json_text,
            articulation_packet_json_text,
            delivery_packet_json_text,
            features_json_text,
            output_matrix_json_text,
            debug_text,
            logs_text,
            chunk_paths,
            reports,
        ) = run_pipeline(
            audio_path,
            process_audio,
            save_audio_fn,
            chunks_dir,
            process_chunks,
            stt_model_name,
            process_timestamps,
            analyze_audio,
            build_and_save_dataset,
            dataset_dir,
            build_coach_packets,
        )

        chunk_updates = []
        for i in range(MAX_DISPLAYED_CHUNKS):
            if i < len(chunk_paths):
                chunk_updates.append(
                    gr.update(value=chunk_paths[i], label=f"chunk_{i + 1}.wav", visible=True)
                )
            else:
                chunk_updates.append(gr.update(value=None, visible=False))

        return (
            summary_text,
            segments_text,
            transcript_text,
            sentences_text,
            words_text,
            fillers_text,
            pauses_text,
            speaking_speed_text,
            pronunciation_summary_text,
            phoneme_analysis_text,
            mispronounced_words_text,
            stress_analysis_text,
            rhythm_analysis_text,
            pronunciation_json_text,
            mti_summary_text,
            mti_vowel_patterns_text,
            mti_consonant_patterns_text,
            mti_stress_transfer_text,
            mti_speech_statistics_text,
            mti_patterns_detected_text,
            mti_json_text,
            intonation_summary_text,
            pitch_analysis_text,
            energy_analysis_text,
            monotonicity_analysis_text,
            emphasis_analysis_text,
            intonation_json_text,
            engagement_summary_text,
            engagement_strengths_text,
            engagement_improvements_text,
            engagement_timeline_text,
            engagement_energy_patterns_text,
            engagement_pause_patterns_text,
            engagement_speaking_dynamics_text,
            engagement_json_text,
            dataset_status_text,
            timeline_json_text,
            articulation_packet_json_text,
            delivery_packet_json_text,
            features_json_text,
            output_matrix_json_text,
            debug_text,
            logs_text,
            *chunk_updates,
            reports,
        )

    def on_generate_feedback_clicked(reports):
        if not reports:
            not_ready = "Run Analysis first — no results to build feedback from yet."
            return not_ready, "", "None.", "None.", "None.", "{}"

        evidence_packet = build_evidence_packet(
            reports["fluency"],
            reports["pronunciation"],
            reports["mti"],
            reports["intonation"],
            reports["engagement"],
            reports["transcript"],
            reports["total_duration"],
        )

        feedback = generate_feedback(evidence_packet)

        return (
            _format_feedback_status(feedback),
            _format_feedback_overall(feedback),
            _format_feedback_strengths(feedback),
            _format_feedback_improvements(feedback),
            _format_feedback_practice_tips(feedback),
            _format_raw_feedback_json(evidence_packet, feedback),
        )

    with gr.Blocks(title="Silero VAD + STT - Manual Testing") as demo:
        gr.Markdown(
            "# Silero VAD + Speech-to-Text — Manual Testing Interface\n"
            "Dev/testing tool only. Record or upload interview audio to run VAD, "
            "generate speech chunks, transcribe them, and inspect the results."
        )

        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
            label="Interview Audio (.wav, .mp3, .m4a)",
        )
        run_button = gr.Button("Run VAD", variant="primary")

        with gr.Row():
            summary_output = gr.Textbox(label="Summary", lines=4, interactive=False)
            debug_output = gr.Textbox(label="Debug Information", lines=4, interactive=False)

        transcript_output = gr.Textbox(label="Transcript", lines=6, interactive=False)
        segments_output = gr.Textbox(label="Speech Segments", lines=14, interactive=False)

        with gr.Row():
            sentences_output = gr.Textbox(label="Sentence Timestamps", lines=16, interactive=False)
            words_output = gr.Textbox(label="Word Timestamps", lines=16, interactive=False)

        gr.Markdown("### Fluency Analysis")
        with gr.Row():
            fillers_output = gr.Textbox(label="Fillers", lines=4, interactive=False)
            pauses_output = gr.Textbox(label="Pauses", lines=4, interactive=False)
            speaking_speed_output = gr.Textbox(label="Speaking Speed", lines=4, interactive=False)

        gr.Markdown(
            "### Pronunciation Analysis\n"
            "Real: phoneme detection (wav2vec2-espeak CTC acoustic model) and rhythm "
            "scoring (word-timing variability) are genuine analysis of the audio. "
            "Stress *detection* is still a placeholder (assumes correct stress; real "
            "pitch/energy analysis isn't integrated yet) — see the PLACEHOLDER comment "
            "in `feature_extractors/audio/pronunciation/stress_placement.py`."
        )
        pronunciation_summary_output = gr.Textbox(
            label="Overall Pronunciation Summary", lines=9, interactive=False
        )
        with gr.Row():
            phoneme_analysis_output = gr.Textbox(label="Phoneme Analysis", lines=14, interactive=False)
            mispronounced_words_output = gr.Textbox(
                label="Mispronounced Words", lines=14, interactive=False
            )
        with gr.Row():
            stress_analysis_output = gr.Textbox(
                label="Stress Placement", lines=14, interactive=False
            )
            rhythm_analysis_output = gr.Textbox(label="Rhythm Analysis", lines=14, interactive=False)

        with gr.Accordion("Raw Pronunciation JSON (debug)", open=False):
            pronunciation_json_output = gr.Textbox(
                label="analyze_pronunciation() output", lines=25, interactive=False
            )

        gr.Markdown(
            "### Mother Tongue Influence (MTI)\n"
            "Goal is interview clarity, not accent classification — this never labels "
            "region/accent/native language, only phoneme-level patterns and their "
            "clarity impact. Vowel and consonant patterns reuse the same real wav2vec2 "
            "phoneme data as Pronunciation Analysis above (no extra model pass); each "
            "phoneme error is classified in exactly one panel. Stress Transfer is "
            "placeholder-dependent — it inherits Pronunciation Analysis's stress-detection "
            "placeholder, so it will read empty until real acoustic stress detection is "
            "integrated there."
        )
        mti_summary_output = gr.Textbox(label="Overall Summary", lines=7, interactive=False)
        with gr.Row():
            mti_vowel_patterns_output = gr.Textbox(label="Vowel Patterns", lines=14, interactive=False)
            mti_consonant_patterns_output = gr.Textbox(
                label="Consonant Patterns", lines=14, interactive=False
            )
        with gr.Row():
            mti_stress_transfer_output = gr.Textbox(label="Stress Transfer", lines=14, interactive=False)
        with gr.Row():
            mti_speech_statistics_output = gr.Textbox(
                label="Speech Statistics", lines=5, interactive=False
            )
            mti_patterns_detected_output = gr.Textbox(
                label="Patterns Detected", lines=5, interactive=False
            )

        with gr.Accordion("Raw MTI JSON (debug)", open=False):
            mti_json_output = gr.Textbox(label="analyze_mti() output", lines=25, interactive=False)

        gr.Markdown(
            "### Intonation Analysis\n"
            "Real: pitch (librosa.yin) and energy (RMS) are genuine signal analysis "
            "of the audio, not placeholders. Word importance for emphasis detection "
            "uses a stopword heuristic, not a real POS tagger — see the comment in "
            "`feature_extractors/audio/intonation/emphasis.py`. All scoring thresholds "
            "are documented starting heuristics, not independently validated yet."
        )
        intonation_summary_output = gr.Textbox(label="Overall Summary", lines=6, interactive=False)
        with gr.Row():
            pitch_analysis_output = gr.Textbox(label="Pitch Analysis", lines=12, interactive=False)
            energy_analysis_output = gr.Textbox(label="Energy Analysis", lines=12, interactive=False)
        with gr.Row():
            monotonicity_analysis_output = gr.Textbox(
                label="Monotonicity Analysis", lines=12, interactive=False
            )
            emphasis_analysis_output = gr.Textbox(label="Emphasis Analysis", lines=12, interactive=False)

        with gr.Accordion("Raw Intonation JSON (debug)", open=False):
            intonation_json_output = gr.Textbox(
                label="analyze_intonation() output", lines=25, interactive=False
            )

        gr.Markdown(
            "### Engagement Analysis\n"
            "Synthesis only — no new audio analysis happens here. Every score below "
            "is built by combining Fluency, Pronunciation, MTI, and Intonation's "
            "already-computed signals (see `feature_extractors/audio/engagement/`), "
            "so this section is fast regardless of recording length."
        )
        engagement_summary_output = gr.Textbox(label="Overall Summary", lines=3, interactive=False)
        with gr.Row():
            engagement_strengths_output = gr.Textbox(label="Strengths", lines=8, interactive=False)
            engagement_improvements_output = gr.Textbox(
                label="Improvement Areas", lines=8, interactive=False
            )
        engagement_timeline_output = gr.Textbox(label="Timeline Analysis", lines=8, interactive=False)
        with gr.Row():
            engagement_energy_patterns_output = gr.Textbox(
                label="Energy Patterns", lines=12, interactive=False
            )
            engagement_pause_patterns_output = gr.Textbox(
                label="Pause Patterns", lines=12, interactive=False
            )
        engagement_speaking_dynamics_output = gr.Textbox(
            label="Speaking Dynamics", lines=14, interactive=False
        )

        with gr.Accordion("Raw Engagement JSON (debug)", open=False):
            engagement_json_output = gr.Textbox(
                label="analyze_engagement() output", lines=25, interactive=False
            )

        gr.Markdown(
            "### Audio Dataset Output Matrix\n"
            "Every session's Fluency/Pronunciation/MTI/Intonation/Engagement run is also "
            "persisted as a two-level dataset record (see `distillation/dataset_builder.py`), "
            "for later synthetic-label generation — not for feedback. **Level 1 (Timeline)** is "
            "objective evidence only (audio metadata, chunks, sentences, words with real "
            "confidence, real acoustic phoneme timestamps, raw pitch/energy contours) — it never "
            "changes even if the analyzers below are later replaced. The **Articulation** and "
            "**Delivery Coach Packets** are the segregated, cleaned, severity-capped model-facing "
            "inputs built from Level 2 (see `feedback/coach_packets.py`) — stress quarantined, "
            "evidence capped top-5 by severity, engagement reduced to its top-line score/level. "
            "**Level 2 (Features)** is the interpretation layer — the five analyzers' full output, "
            "unchanged in shape from the panels above."
        )
        dataset_status_output = gr.Textbox(label="Dataset Status", lines=4, interactive=False)
        with gr.Accordion("Level 1 — Timeline Data", open=False):
            timeline_json_output = gr.Textbox(
                label="Audio metadata, chunks, sentences, words, detected phonemes, acoustic contours",
                lines=25, interactive=False,
            )
        with gr.Accordion("Articulation Coach Packet", open=False):
            articulation_packet_json_output = gr.Textbox(
                label="Pronunciation + MTI, segregated/cleaned/capped (build_articulation_packet)",
                lines=25, interactive=False,
            )
        with gr.Accordion("Delivery Coach Packet", open=False):
            delivery_packet_json_output = gr.Textbox(
                label="Fluency + Intonation + Engagement + rhythm, segregated/cleaned/capped (build_delivery_packet)",
                lines=25, interactive=False,
            )
        with gr.Accordion("Level 2 — Feature Data", open=False):
            features_json_output = gr.Textbox(
                label="Fluency, Pronunciation, MTI, Intonation, Engagement (full analyzer output)",
                lines=25, interactive=False,
            )
        with gr.Accordion("Complete Output Matrix (debug)", open=False):
            output_matrix_json_output = gr.Textbox(
                label="{\"timeline\": ..., \"features\": ...} — fully assembled, copy-friendly",
                lines=25, interactive=False,
            )

        reports_state = gr.State({})

        gr.Markdown(
            "### AI Feedback (LLM)\n"
            "Separate step, not run automatically — click below after Run Analysis "
            "completes. Sends a compact evidence packet (scores + observations + "
            "timestamps from the five sections above; raw stress fields excluded "
            "since stress detection is still a placeholder) to a single LLM call "
            "for narrative coaching feedback. No fine-tuning/distillation involved — "
            "prompt-templated only. Requires `ZIQRA_LLM_API_KEY` to be set."
        )
        feedback_button = gr.Button("Generate AI Feedback")
        feedback_status_output = gr.Textbox(label="Status", lines=1, interactive=False)
        feedback_overall_output = gr.Textbox(label="Overall Assessment", lines=5, interactive=False)
        with gr.Row():
            feedback_strengths_output = gr.Textbox(label="Key Strengths", lines=8, interactive=False)
            feedback_improvements_output = gr.Textbox(
                label="Priority Improvements", lines=8, interactive=False
            )
        feedback_practice_tips_output = gr.Textbox(label="Practice Tips", lines=6, interactive=False)

        with gr.Accordion("Raw LLM JSON (debug)", open=False):
            feedback_json_output = gr.Textbox(
                label="evidence_packet + generate_feedback() output", lines=25, interactive=False
            )

        gr.Markdown("### Generated Chunks")
        chunk_players = [
            gr.Audio(label=f"chunk_{i + 1}.wav", visible=False, interactive=False)
            for i in range(MAX_DISPLAYED_CHUNKS)
        ]

        logs_output = gr.Textbox(label="Logs", lines=10, interactive=False)

        run_button.click(
            fn=on_run_clicked,
            inputs=[audio_input],
            outputs=[
                summary_output,
                segments_output,
                transcript_output,
                sentences_output,
                words_output,
                fillers_output,
                pauses_output,
                speaking_speed_output,
                pronunciation_summary_output,
                phoneme_analysis_output,
                mispronounced_words_output,
                stress_analysis_output,
                rhythm_analysis_output,
                pronunciation_json_output,
                mti_summary_output,
                mti_vowel_patterns_output,
                mti_consonant_patterns_output,
                mti_stress_transfer_output,
                mti_speech_statistics_output,
                mti_patterns_detected_output,
                mti_json_output,
                intonation_summary_output,
                pitch_analysis_output,
                energy_analysis_output,
                monotonicity_analysis_output,
                emphasis_analysis_output,
                intonation_json_output,
                engagement_summary_output,
                engagement_strengths_output,
                engagement_improvements_output,
                engagement_timeline_output,
                engagement_energy_patterns_output,
                engagement_pause_patterns_output,
                engagement_speaking_dynamics_output,
                engagement_json_output,
                dataset_status_output,
                timeline_json_output,
                articulation_packet_json_output,
                delivery_packet_json_output,
                features_json_output,
                output_matrix_json_output,
                debug_output,
                logs_output,
                *chunk_players,
                reports_state,
            ],
        )

        feedback_button.click(
            fn=on_generate_feedback_clicked,
            inputs=[reports_state],
            outputs=[
                feedback_status_output,
                feedback_overall_output,
                feedback_strengths_output,
                feedback_improvements_output,
                feedback_practice_tips_output,
                feedback_json_output,
            ],
        )

    demo.launch(server_name=host, server_port=port, inbrowser=True, share=False)
