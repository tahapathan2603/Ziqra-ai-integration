"""
Per-coach evidence packets: the compressed, segregated model-facing inputs for
the two audio feedback student models (Articulation Coach, Delivery Coach).

This is the "narrate, don't score" boundary in serialized form. Like
evidence_packet.py, a packet is the ONLY thing a coach model sees, so the model
narrates decisions the deterministic feature extractors already made rather than
re-deriving anything. Unlike evidence_packet.py (one blob for the current
single-LLM feedback path, left untouched here), this splits the five analyzer
reports along the segmental/suprasegmental seam into two self-contained packets:

  - Articulation packet <- Pronunciation + MTI (phoneme-level accuracy)
  - Delivery packet     <- Fluency + Intonation + Engagement (prosody/timing),
                           plus Pronunciation's rhythm (rhythm is timing, not
                           articulation, so it rides with Delivery).

Three cleaning rules are enforced here, per the audio-model architecture plan
(~/.claude/plans/declarative-gathering-sutton.md). Each exists to stop a model
from being taught something false or double-counted:

  1. NO stress signals anywhere. stress_placement.py is a placeholder that
     always reports 100% / no errors (see pronunciation_analyzer.py's note and
     evidence_packet.py). Feeding it in teaches confident, false praise about
     stress that was never measured. Quarantined until real acoustic stress
     detection lands. Covers pronunciation stress AND MTI stress_transfer.
     In practice stress_errors/stress_transfer are always empty today, so the
     structured lists carry no stress; the label filter below is defensive, in
     case real stress detection lands before this quarantine is lifted.
  2. Engagement contributes ONLY its top-line score + level, as an
     executive-summary seed. Its re-derived internals (energy_patterns,
     speaking_dynamics, per-window timeline, and the synthesized strengths/
     improvement_areas) are recomputations of the exact Fluency/Intonation
     evidence the Delivery packet already carries — feeding both double-counts
     and re-opens the self-contradiction class fixed in engagement_analyzer.py.
  3. Emphasis (intonation.emphasis / under_emphasized_words) is reliability-
     gated OUT: its under-emphasis detection is a documented, unvalidated
     heuristic. The intonation SCORE (which emphasis feeds) still passes through
     untouched — we just don't hand the model emphasis's word-level claims.

Compression (plan Part 4): every evidence list is capped to the top
TOP_K_EVIDENCE items by severity; contours and full word/phoneme dumps never
enter a packet (pitch/energy become summary stats). The on-disk Level 1/2
dataset still preserves everything — these packets are the model-facing view,
not the archive.
"""

from typing import Dict, List, Optional

from backend.feature_extractors.audio.phoneme_patterns import severity_rank

# Keep only the N highest-severity items per evidence list. Tighter than
# evidence_packet.py's 10: a coach narrates a few concrete, prioritized issues,
# and a long tail of low-severity items dilutes the signal the model learns.
TOP_K_EVIDENCE = 5

# Substrings that mark a free-text observation (or issue label) as carrying a
# quarantined/reliability-gated signal, so it can be dropped from the rendered
# observations lists too — not just the structured fields. "stress" covers
# cleaning rule 1 (placeholder stress); "emphasi" covers rule 3 (emphasis).
# In practice the stress observation lines never render today (they require
# non-empty stress_errors, impossible with the placeholder), so that filter is
# defensive; the emphasis line genuinely appears and must be dropped.
_STRESS_MARKERS = ("stress",)
_EMPHASIS_MARKERS = ("emphasi",)


def _round_ts(value):
    return round(value, 2) if isinstance(value, (int, float)) else value


def _drop_observations_mentioning(observations: List[str], markers) -> List[str]:
    """Drop any observation string containing one of `markers` (case-insensitive),
    so a reliability-gated/quarantined signal can't slip back in as prose."""
    return [
        obs
        for obs in observations
        if not any(marker in obs.lower() for marker in markers)
    ]


def _top_by_severity(items: List[Dict], k: int = TOP_K_EVIDENCE) -> List[Dict]:
    """Highest-severity-first, truncated to k. Items with no severity field
    rank last (severity_rank returns 0 for unknown/empty)."""
    if not items:
        return []
    ranked = sorted(items, key=lambda i: severity_rank(i.get("severity", "")), reverse=True)
    return ranked[:k]


def _is_stress_label(label: Optional[str]) -> bool:
    return bool(label) and any(marker in label.lower() for marker in _STRESS_MARKERS)


def build_articulation_packet(pronunciation_report: Dict, mti_report: Dict) -> Dict:
    """
    Segmental / phoneme-level packet: Pronunciation + MTI only. No stress, no
    rhythm (rhythm goes to Delivery). See module docstring for the cleaning
    rules applied.

    Args:
        pronunciation_report: analyze_pronunciation() output.
        mti_report: analyze_mti() output.

    Returns: the Articulation Coach's evidence packet.
    """
    mispronounced = _top_by_severity(pronunciation_report.get("mispronounced_words", []))

    # Vowel + consonant patterns only — MTI's stress_transfer is quarantined
    # (rule 1). This mirrors evidence_packet._mti_patterns, which also excludes
    # stress_transfer.
    patterns = _top_by_severity(
        list(mti_report.get("vowel_patterns", [])) + list(mti_report.get("consonant_patterns", []))
    )

    # Defensively drop any stress-labeled entries from MTI's precomputed
    # label/summary fields (rule 1).
    patterns_detected = [p for p in mti_report.get("patterns_detected", []) if not _is_stress_label(p)]
    top_recurring_issue = mti_report.get("top_recurring_issue")
    if _is_stress_label(top_recurring_issue):
        top_recurring_issue = None

    return {
        "coach": "articulation",
        "pronunciation": {
            "score": pronunciation_report.get("pronunciation_score"),
            "phoneme_accuracy_pct": round(pronunciation_report.get("phoneme_accuracy", 0) * 100),
            "observations": _drop_observations_mentioning(
                pronunciation_report.get("overall_observations", []), _STRESS_MARKERS
            ),
            "mispronounced_words": [
                {
                    "word": w["word"],
                    "severity": w["severity"],
                    "start": _round_ts(w.get("start")),
                    "end": _round_ts(w.get("end")),
                }
                for w in mispronounced
            ],
        },
        "mti": {
            "score": mti_report.get("overall_mti_score"),
            "clarity_impact": mti_report.get("clarity_impact"),
            "top_recurring_issue": top_recurring_issue,
            "patterns_detected": patterns_detected,
            "clarity_risk_words": list(mti_report.get("clarity_risk_words", []))[:TOP_K_EVIDENCE],
            "observations": _drop_observations_mentioning(
                mti_report.get("overall_observations", []), _STRESS_MARKERS
            ),
            "patterns": [
                {
                    "word": p["word"],
                    "issue": p["issue"],
                    "expected_phoneme": p["expected"],
                    "detected_phoneme": p["detected"],
                    "severity": p["severity"],
                    "start": _round_ts(p.get("timestamp", {}).get("start")),
                    "end": _round_ts(p.get("timestamp", {}).get("end")),
                }
                for p in patterns
            ],
        },
    }


def build_delivery_packet(
    fluency_report: Dict,
    intonation_report: Dict,
    engagement_report: Dict,
    pronunciation_report: Dict,
) -> Dict:
    """
    Suprasegmental / prosody-and-timing packet: Fluency + Intonation +
    Engagement (top-line only), plus Pronunciation's rhythm. Emphasis is
    reliability-gated out; engagement internals are dropped. See module
    docstring for the cleaning rules applied.

    Args:
        fluency_report: analyze_fluency() output.
        intonation_report: analyze_intonation() output.
        engagement_report: analyze_engagement() output.
        pronunciation_report: analyze_pronunciation() output — used ONLY for
            rhythm_score / rhythm_issues (timing regularity is prosody).

    Returns: the Delivery Coach's evidence packet.
    """
    fillers = fluency_report.get("fillers", {})
    pauses = fluency_report.get("pauses", {}).get("pauses", [])
    speaking_speed = fluency_report.get("speaking_speed", {})

    # Monotone + low-energy sections, severity-ranked. Emphasis's
    # under_emphasized_words are deliberately excluded (rule 3); the intonation
    # score already reflects emphasis.
    monotone = [
        {**s, "kind": "monotone"}
        for s in intonation_report.get("monotonicity", {}).get("monotone_sections", [])
    ]
    low_energy = [
        {**s, "kind": "low_energy"}
        for s in intonation_report.get("energy_variation", {}).get("low_energy_sections", [])
    ]
    flat_sections = _top_by_severity(monotone + low_energy)

    return {
        "coach": "delivery",
        "fluency": {
            "speaking_speed": {
                "words_per_minute": speaking_speed.get("words_per_minute"),
                "classification": speaking_speed.get("classification"),
            },
            "filler_usage": {
                "count": fillers.get("filler_count", 0),
                "per_minute": fillers.get("fillers_per_minute", 0.0),
                "classification": fillers.get("classification"),
                "examples": [f["word"] for f in fillers.get("fillers", [])][:TOP_K_EVIDENCE],
            },
            # Collapsed pause taxonomy: counts by type, not every pause object
            # (dead_air / hesitation are subtypes of pause, so one taxonomy).
            "pauses": {
                "total": len(pauses),
                "dead_air_count": sum(1 for p in pauses if p.get("type") == "dead_air"),
                "long_pause_count": sum(1 for p in pauses if p.get("type") == "long_pause"),
                "hesitation_count": sum(1 for p in pauses if p.get("type") == "hesitation"),
            },
        },
        "intonation": {
            "score": intonation_report.get("intonation_score"),
            "delivery_label": intonation_report.get("delivery_label"),
            # Emphasis is reliability-gated (rule 3), so drop its rendered
            # observation line too — not just the structured emphasis block.
            "observations": _drop_observations_mentioning(
                intonation_report.get("overall_observations", []), _EMPHASIS_MARKERS
            ),
            # Summary stats only — the raw pitch/energy contours never enter a
            # packet (they live in the Level 1 dataset).
            "pitch": {
                "average_hz": intonation_report.get("pitch_variation", {}).get("average_pitch"),
                "range_hz": intonation_report.get("pitch_variation", {}).get("pitch_range"),
            },
            "flat_sections": [
                {
                    "kind": s["kind"],
                    "start": _round_ts(s.get("start")),
                    "end": _round_ts(s.get("end")),
                    "severity": s.get("severity"),
                }
                for s in flat_sections
            ],
        },
        # Rhythm is timing regularity -> prosody, so it rides with Delivery even
        # though pronunciation_analyzer.py is what computes it.
        "rhythm": {
            "score_pct": round(pronunciation_report.get("rhythm_score", 0) * 100),
            "issues": pronunciation_report.get("rhythm_issues", []),
        },
        # Executive-summary seed ONLY (rule 2) — no re-derived internals.
        "engagement": {
            "score": engagement_report.get("engagement_score"),
            "level": engagement_report.get("engagement_level"),
        },
    }


def build_coach_packets(analysis: Dict, session_id: Optional[str] = None) -> Dict:
    """
    Segregate one session's analysis into both per-coach packets.

    Args:
        analysis: a dict carrying the five analyzer reports directly under the
            keys "fluency", "pronunciation", "mti", "intonation", "engagement".
            This is exactly the shape of both audio_analyzer.analyze_audio()'s
            return AND a Level 2 features file's "analysis" block
            (dataset/features/session_*.json), so either can be passed in as-is.
        session_id: optional linkage key to stamp onto the result (e.g. when
            building packets from a stored features file for teacher labeling).

    Returns:
        {"articulation": {...}, "delivery": {...}, ["session_id": str]}
    """
    packets = {
        "articulation": build_articulation_packet(analysis["pronunciation"], analysis["mti"]),
        "delivery": build_delivery_packet(
            analysis["fluency"],
            analysis["intonation"],
            analysis["engagement"],
            analysis["pronunciation"],
        ),
    }
    if session_id is not None:
        packets["session_id"] = session_id
    return packets
