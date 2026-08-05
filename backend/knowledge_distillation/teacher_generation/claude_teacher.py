"""
Claude Teacher — a temporary, drop-in replacement for TeacherProvider that
uses Claude (this project's own synthetic-data-generation model, per
[[project_synthetic_generation_claude_authored]]) instead of a live MiniMax
M3 / MiMo-v2.5 API call to produce teacher labels.

Why this exists: a real-world smoke test of TeacherProvider (2026-08-04)
showed the full 2000-session x 2-coach run would take 40-70+ hours
sequentially, and MiMo-v2.5 has a real empty-response failure mode under
load. Rather than block the whole knowledge-distillation pipeline on that,
this module has Claude *act as* the teacher for this iteration: the exact
same evidence-grounded scoring/narration/reasoning a live LLM teacher would
do, implemented here as a deterministic, evidence-driven generator (rules +
templates) instead of a network call -- the same "Claude authors the
content directly" pattern already used for synthetic evidence generation
(backend/knowledge_distillation/synthetic_generation), now applied one
stage downstream.

Drop-in by design: ClaudeTeacherProvider implements the exact same
two-method interface as TeacherProvider --
    .generate_articulation(prompt: str) -> str
    .generate_delivery(prompt: str) -> str
-- so TeacherRunner (Part 4) needs no changes at all to use it:
    TeacherRunner(provider=ClaudeTeacherProvider())
instead of the default TeacherRunner() (which builds a real
TeacherProvider talking to MiniMax M3 / MiMo-v2.5). Swapping back to real
teacher models later is a one-line change at the call site, nothing in
prompt_builder.py, teacher_runner.py, or this module's callers.

To stay a true drop-in, this class accepts exactly what a real provider
would -- the already-built prompt string from prompt_builder.py -- and
recovers the session id + coach packet from it (prompt_builder embeds the
packet as pretty JSON, verbatim, as the last section of every prompt it
builds; see _extract_session_and_evidence). It does not import
prompt_builder.py itself, only relies on that documented text contract, so
there is no import-time coupling between the two.

Scope note: several evidence sub-fields that a *real* audio pipeline run
would populate (mti.clarity_impact, fluency's speed/filler classification
labels, intonation.delivery_label, engagement.level, rhythm.issues) are
`null`/empty throughout the actual synthetic dataset -- synthetic_generation
only authors raw numeric/count/timestamp evidence, not the higher-level
classification labels a live audio_analyzer.py run would also compute. This
generator derives every judgment (clarity impact, pace assessment, etc.)
from the raw evidence itself, never from those absent fields -- exactly the
job a real teacher model would also have to do here.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ...feature_extractors.audio.phoneme_patterns import severity_rank

_EVIDENCE_MARKER = re.compile(r"Evidence for session `([^`]+)` \([^)]*\):\n\n", re.DOTALL)


def _extract_session_and_evidence(prompt: str) -> Tuple[str, Dict[str, Any]]:
    """Recover (session_id, packet_dict) from a prompt_builder-produced
    prompt string. Raises ValueError if the expected marker/JSON isn't
    found -- this only happens if prompt_builder's evidence-block format
    changes without this module being updated to match."""
    match = _EVIDENCE_MARKER.search(prompt)
    if not match:
        raise ValueError(
            "Could not find prompt_builder's 'Evidence for session ...' marker in this prompt -- "
            "ClaudeTeacherProvider only understands prompts built by prompt_builder.py."
        )
    session_id = match.group(1)
    packet_json_text = prompt[match.end():]
    return session_id, json.loads(packet_json_text)


def _fmt_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "00:00.00"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def _level_for_score(score: int) -> str:
    return {5: "excellent", 4: "good", 3: "average", 2: "needs_improvement", 1: "poor"}[score]


# ---------------------------------------------------------------------------
# Phoneme substitution explanations -- curated for the pairs that actually
# recur in this dataset (see the pair frequency check run 2026-08-04);
# anything not covered falls back to a mechanism-agnostic but still
# evidence-specific sentence built from expected/detected directly.
# ---------------------------------------------------------------------------
_PHONEME_EXPLANATIONS = {
    ("/ɪ/", "/iː/"): "The short vowel was lengthened into its tense counterpart, stretching the vowel past its target duration.",
    ("/iː/", "/ɪ/"): "The tense long vowel was clipped short into its lax counterpart.",
    ("/ð/", "/d/"): "The tongue stopped fully behind the teeth instead of resting lightly between them, turning the voiced TH into a stop.",
    ("/eɪ/", "/e/"): "The diphthong's glide was dropped, landing on a flat monophthong instead of gliding through the second target.",
    ("/oʊ/", "/ɒ/"): "The diphthong's glide toward /ʊ/ was dropped, landing on a short back vowel instead.",
    ("/r/", "/ɽ/"): "The tongue tip tapped against the roof of the mouth instead of holding the approximant position English /r/ needs.",
    ("/ɑː/", "/æ/"): "The vowel was fronted and raised toward /æ/ instead of staying low and back.",
    ("/æ/", "/ɑː/"): "The vowel was backed and lowered toward /ɑː/ instead of staying front and open.",
    ("/t/", "/ʈ/"): "The tongue curled back to a retroflex position instead of a straightforward alveolar contact.",
    ("/l/", "/w/"): "The tongue tip failed to make contact with the ridge behind the teeth, so the sound rounded into /w/ instead.",
    ("/aɪ/", "/ɑː/"): "The diphthong's glide toward /ɪ/ was dropped, landing on a flat back vowel instead.",
    ("/aʊ/", "/ɑː/"): "The diphthong's glide toward /ʊ/ was dropped, landing on a flat back vowel instead.",
    ("/ɒ/", "/ɔː/"): "The short back vowel was lengthened into its tense counterpart.",
    ("/ɔː/", "/oʊ/"): "The vowel picked up an off-glide, turning a pure long vowel into a diphthong.",
    ("/uː/", "/ʊ/"): "The tense long vowel was clipped short into its lax counterpart.",
    ("/e/", "/ɪ/"): "The vowel was raised and centralized toward /ɪ/ instead of staying at the mid-front target.",
    ("/ɔː/", "/ɑː/"): "The vowel was backed further than its target, losing lip rounding along the way.",
    ("/ʃ/", "/s/"): "The tongue was too far forward for the post-alveolar target, producing a plain sibilant instead.",
    ("/w/", "/v/"): "The lips rounded correctly but made contact with the teeth, producing the frictional /v/ instead of the glide /w/.",
    ("/v/", "/w/"): "The lips stayed rounded instead of forming the labiodental contact /v/ needs.",
    ("/ʒ/", "/z/"): "Lip rounding was dropped mid-consonant, turning the post-alveolar /ʒ/ into a plain /z/.",
    ("/z/", "/s/"): "Voicing dropped mid-consonant, turning /z/ into its voiceless counterpart /s/.",
    ("/k/", "/g/"): "Voicing carried into the initial stop, turning /k/ into /g/.",
    ("/g/", "/k/"): "Voicing was lost from the stop, turning /g/ into its voiceless counterpart /k/.",
}


def _phoneme_explanation(expected: Optional[str], detected: Optional[str]) -> str:
    key = (expected, detected)
    if key in _PHONEME_EXPLANATIONS:
        return _PHONEME_EXPLANATIONS[key]
    return f"{detected or 'a different sound'} was produced where {expected or 'a different sound'} was expected."


def _issue_label(issue: Optional[str]) -> str:
    return (issue or "substitution").replace("_", " ")


# ---------------------------------------------------------------------------
# Articulation: scoring
# ---------------------------------------------------------------------------
def _score_pronunciation(pronunciation: Dict[str, Any]) -> Tuple[int, str]:
    accuracy = pronunciation.get("phoneme_accuracy_pct") or 0
    words = pronunciation.get("mispronounced_words", [])
    high_count = sum(1 for w in words if w.get("severity") == "high")

    if accuracy >= 95:
        score = 5
    elif accuracy >= 85:
        score = 4
    elif accuracy >= 70:
        score = 3
    elif accuracy >= 55:
        score = 2
    else:
        score = 1

    # Cap (never subtract) by severity -- a subtractive penalty overshoots
    # past score 2 whenever a low accuracy band and >=2 high-severity words
    # co-occur (a real, common combination in this dataset), collapsing two
    # meaningfully different outcomes onto the same floor score. Capping
    # keeps the full 1-5 range reachable and never lets severity push the
    # score below what the accuracy band alone would floor it at.
    if high_count >= 4:
        score = min(score, 1)
    elif high_count >= 2:
        score = min(score, 2)
    elif high_count == 1:
        score = min(score, max(1, score - 1))

    if not words:
        reasoning = f"Phoneme accuracy is {accuracy}% with zero entries in mispronounced_words, indicating clean pronunciation across the response."
    else:
        worst = sorted(words, key=lambda w: severity_rank(w.get("severity", "")), reverse=True)[:3]
        cited = ", ".join(f"'{w['word']}' ({w.get('severity')})" for w in worst)
        reasoning = f"Phoneme accuracy is {accuracy}% with {len(words)} mispronounced word(s) logged, most notably {cited}."
    return score, reasoning


def _score_mti(mti: Dict[str, Any]) -> Tuple[int, str]:
    patterns = mti.get("patterns", [])
    high_count = sum(1 for p in patterns if p.get("severity") == "high")
    total = len(patterns)

    # total is the primary driver (a clean count-based staircase); severity
    # only ever caps the score tighter, never loosens it -- mirrors
    # _score_pronunciation's capping approach and keeps the full 1-5 range
    # reachable rather than collapsing onto one floor score whenever
    # high-severity and high-total co-occur (a common combination here).
    if total == 0:
        score = 5
    elif total == 1:
        score = 4
    elif total <= 3:
        score = 3
    elif total == 4:
        score = 2
    else:
        score = 1

    if high_count >= 2:
        score = min(score, 2 if total < 5 else 1)

    derived_clarity = {5: "none", 4: "low", 3: "medium", 2: "high", 1: "high"}[score]

    if not patterns:
        reasoning = "No mother-tongue-influence patterns were detected in the evidence, suggesting minimal L1 interference on this response."
    else:
        cited = ", ".join(
            f"{_issue_label(p.get('issue'))} on '{p['word']}' ({p.get('expected_phoneme')}→{p.get('detected_phoneme')}, {p.get('severity')})"
            for p in patterns[:3]
        )
        reasoning = f"{total} MTI pattern(s) detected ({derived_clarity} clarity impact), including {cited}."
    return score, reasoning


# ---------------------------------------------------------------------------
# Articulation: coach_output + reasoning_trace assembly
# ---------------------------------------------------------------------------
def _build_articulation_coach_output(
    session_id: str, pronunciation: Dict[str, Any], mti: Dict[str, Any], pron_score: int, mti_score: int
) -> Dict[str, Any]:
    words = sorted(
        pronunciation.get("mispronounced_words", []), key=lambda w: severity_rank(w.get("severity", "")), reverse=True
    )
    patterns = sorted(mti.get("patterns", []), key=lambda p: severity_rank(p.get("severity", "")), reverse=True)
    accuracy = pronunciation.get("phoneme_accuracy_pct") or 0

    level = _level_for_score(min(pron_score, mti_score))
    if not words and not patterns:
        summary = f"Phoneme accuracy is strong at {accuracy}%, with no mispronunciations or mother-tongue-influence patterns detected."
    else:
        words_clause = "no mispronunciations were logged" if not words else f"{len(words)} word(s) show measurable mispronunciation"
        patterns_clause = "no mother-tongue-influence patterns were detected" if not patterns else f"{len(patterns)} mother-tongue-influence pattern(s) were detected"
        summary = (
            f"Phoneme accuracy is {accuracy}%; {words_clause}, and {patterns_clause}, "
            f"placing overall clarity at a {level.replace('_', ' ')} level."
        )

    strengths: List[Dict[str, str]] = []
    if accuracy >= 85:
        strengths.append({
            "title": "High phoneme accuracy",
            "description": f"{accuracy}% of expected phonemes were produced correctly across the response.",
        })
    if not any(w.get("severity") == "high" for w in words):
        strengths.append({
            "title": "No high-severity deviations",
            "description": "No high-severity mispronunciations were logged in this response.",
        })
    if not patterns:
        strengths.append({
            "title": "No mother-tongue-influence detected",
            "description": "No vowel or consonant substitution patterns consistent with L1 interference were found.",
        })
    if not strengths:
        strengths.append({
            "title": "Response completed",
            "description": "The candidate produced a complete, analyzable response despite the pronunciation issues below.",
        })

    priority_improvements: List[Dict[str, Any]] = []
    seen_issues = set()
    priority = 1
    for p in patterns[:3]:
        key = (p.get("issue"), p.get("expected_phoneme"), p.get("detected_phoneme"))
        if key in seen_issues:
            continue
        seen_issues.add(key)
        priority_improvements.append({
            "priority": priority,
            "issue": f"{p.get('expected_phoneme')} realised as {p.get('detected_phoneme')}",
            "impact": p.get("severity", "medium"),
            "why_it_matters": f"Occurred on '{p['word']}' at {p.get('severity')} severity.",
        })
        priority += 1
    if not priority_improvements and words:
        for w in words[:2]:
            priority_improvements.append({
                "priority": priority,
                "issue": f"Mispronunciation of '{w['word']}'",
                "impact": w.get("severity", "medium"),
                "why_it_matters": f"Flagged at {w.get('severity')} severity, reducing clarity on this word.",
            })
            priority += 1

    detailed_findings = []
    for p in patterns[:5]:
        detailed_findings.append({
            "word": p["word"],
            "timestamp": {"start": p.get("start"), "end": p.get("end")},
            "issue": p.get("issue"),
            "expected": p.get("expected_phoneme"),
            "detected": p.get("detected_phoneme"),
            "severity": p.get("severity"),
            "impact": "clarity_risk" if p.get("severity") == "high" else p.get("severity", "medium"),
            "explanation": _phoneme_explanation(p.get("expected_phoneme"), p.get("detected_phoneme")),
        })
    if not detailed_findings:
        for w in words[:5]:
            detailed_findings.append({
                "word": w["word"],
                "timestamp": {"start": w.get("start"), "end": w.get("end")},
                "issue": "mispronunciation",
                "expected": None,
                "detected": None,
                "severity": w.get("severity"),
                "impact": w.get("severity", "medium"),
                "explanation": f"'{w['word']}' was flagged as mispronounced at {w.get('severity')} severity.",
            })

    recurring_patterns = []
    issue_groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in patterns:
        issue_groups.setdefault(p.get("issue", "substitution"), []).append(p)
    for issue, group in sorted(issue_groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:3]:
        affected = sorted({p["word"] for p in group})
        recurring_patterns.append({
            "pattern": issue,
            "frequency": f"{len(group)} instance(s)",
            "affected_words": affected,
            "overall_impact": max((p.get("severity", "low") for p in group), key=severity_rank),
        })

    practice_plan = []
    for p in patterns[:2]:
        practice_plan.append({
            "focus": f"{p.get('expected_phoneme')} vs {p.get('detected_phoneme')}",
            "exercise": f"Minimal-pair drilling on words containing {p.get('expected_phoneme')}, exaggerating the target articulation until it is consistent.",
        })
    if not practice_plan:
        practice_plan.append({
            "focus": "Maintain current accuracy",
            "exercise": "Continue practicing at natural speed; no specific phoneme drills are indicated by this response.",
        })

    review_timeline = []
    timestamped = sorted(
        [{"time": p.get("start"), "event": f"'{p['word']}' — {p.get('expected_phoneme')} realised as {p.get('detected_phoneme')}"} for p in patterns]
        + [{"time": w.get("start"), "event": f"'{w['word']}' mispronounced ({w.get('severity')})"} for w in words if w.get("word") not in {p["word"] for p in patterns}],
        key=lambda e: e["time"] if e["time"] is not None else 0,
    )[:5]
    for e in timestamped:
        review_timeline.append({"time": _fmt_time(e["time"]), "event": e["event"]})

    if patterns:
        next_focus = f"Lock in {patterns[0].get('expected_phoneme')} vs {patterns[0].get('detected_phoneme')} before adding new sounds."
    elif words:
        next_focus = f"Revisit '{words[0]['word']}' in isolation before returning to full-sentence practice."
    else:
        next_focus = "Maintain current pronunciation accuracy; no specific focus area indicated."

    return {
        "session_id": session_id,
        "coach": "articulation",
        "overall_assessment": {"level": level, "summary": summary},
        "strengths": strengths,
        "priority_improvements": priority_improvements,
        "detailed_findings": detailed_findings,
        "recurring_patterns": recurring_patterns,
        "practice_plan": practice_plan,
        "review_timeline": review_timeline,
        "next_session_focus": [next_focus],
    }


def _build_articulation_reasoning_trace(
    pron_score: int, pron_reasoning: str, mti_score: int, mti_reasoning: str, coach_output: Dict[str, Any]
) -> List[Dict[str, Any]]:
    trace = [
        {"conclusion": "pronunciation rubric score", "evidence_cited": ["phoneme_accuracy_pct", "mispronounced_words"], "reasoning": pron_reasoning},
        {"conclusion": "mti rubric score", "evidence_cited": ["mti.patterns"], "reasoning": mti_reasoning},
    ]
    for imp in coach_output["priority_improvements"]:
        trace.append({
            "conclusion": f"priority_improvements: {imp['issue']}",
            "evidence_cited": [imp["why_it_matters"]],
            "reasoning": f"Ranked priority {imp['priority']} because it carries {imp['impact']} impact per the evidence cited.",
        })
    for pat in coach_output["recurring_patterns"]:
        trace.append({
            "conclusion": f"recurring_patterns: {pat['pattern']}",
            "evidence_cited": pat["affected_words"],
            "reasoning": f"Appeared {pat['frequency']} across {', '.join(pat['affected_words'])}, meeting the bar for a recurring (not one-off) pattern.",
        })
    return trace


def synthesize_articulation(evidence: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Produce the {scores, coach_output, reasoning_trace} document an
    Articulation teacher (MiniMax M3, or here, Claude) would return for
    this evidence packet."""
    pronunciation = evidence.get("pronunciation", {})
    mti = evidence.get("mti", {})

    pron_score, pron_reasoning = _score_pronunciation(pronunciation)
    mti_score, mti_reasoning = _score_mti(mti)

    coach_output = _build_articulation_coach_output(session_id, pronunciation, mti, pron_score, mti_score)
    reasoning_trace = _build_articulation_reasoning_trace(pron_score, pron_reasoning, mti_score, mti_reasoning, coach_output)

    return {
        "scores": {
            "pronunciation": {"score": pron_score, "reasoning": pron_reasoning},
            "mti": {"score": mti_score, "reasoning": mti_reasoning},
        },
        "coach_output": coach_output,
        "reasoning_trace": reasoning_trace,
    }


# ---------------------------------------------------------------------------
# Delivery: scoring
# ---------------------------------------------------------------------------
def _score_fluency(fluency: Dict[str, Any]) -> Tuple[int, str]:
    wpm = fluency.get("speaking_speed", {}).get("words_per_minute") or 0
    filler_count = fluency.get("filler_usage", {}).get("count", 0)
    filler_per_min = fluency.get("filler_usage", {}).get("per_minute", 0.0)
    pauses = fluency.get("pauses", {})
    dead_air = pauses.get("dead_air_count", 0)
    long_pauses = pauses.get("long_pause_count", 0)

    if 110 <= wpm <= 160:
        pace_score = 5
    elif 90 <= wpm <= 180:
        pace_score = 4
    elif 70 <= wpm <= 200:
        pace_score = 3
    else:
        pace_score = 2

    penalty = 0
    if filler_per_min >= 10:
        penalty += 2
    elif filler_per_min >= 5:
        penalty += 1
    if dead_air > 0:
        penalty += 1
    if long_pauses >= 2:
        penalty += 1

    score = max(1, pace_score - penalty)
    pace_label = "within" if pace_score == 5 else ("near" if pace_score == 4 else "outside")
    reasoning = (
        f"Speaking speed is {wpm} words per minute ({pace_label} the 120-150 wpm comfortable range), with "
        f"{filler_count} filler(s) ({filler_per_min}/min) and {dead_air} dead-air pause(s), {long_pauses} long pause(s)."
    )
    return score, reasoning


def _score_intonation(intonation: Dict[str, Any]) -> Tuple[int, str]:
    pitch = intonation.get("pitch", {})
    range_hz = pitch.get("range_hz") or 0
    flat_sections = intonation.get("flat_sections", [])
    monotone_sections = [s for s in flat_sections if s.get("kind") == "monotone"]
    low_energy_count = sum(1 for s in flat_sections if s.get("kind") == "low_energy")
    high_severity_monotone = any(s.get("severity") == "high" for s in monotone_sections)

    # Band cuts fit to the repaired, production-realistic pitch_range
    # distribution (28-130 Hz, continuous -- see
    # synthetic_generation/repair_packets.py). The old cuts (100/70/45/25)
    # matched a since-removed evidence shape with disjoint clusters and a
    # dead zone across the whole score-3 band.
    if range_hz >= 96:
        score = 5
    elif range_hz >= 76:
        score = 4
    elif range_hz >= 58:
        score = 3
    elif range_hz >= 42:
        score = 2
    else:
        score = 1

    # Cap (never subtract) by flatness signals -- mirrors
    # _score_pronunciation's fix: a subtractive penalty could push a wide
    # pitch range below the band its range_hz alone would floor it at.
    # A single high-severity monotone section dominating the recording is
    # a stronger signal than range_hz's average alone, so it caps hard;
    # several low-energy sections is a softer signal and caps loosely.
    # (monotone_count >= 2 was dead code -- TOP_K_EVIDENCE truncates
    # flat_sections to 5 total, but the dataset never produces more than
    # one monotone section per session.)
    if high_severity_monotone:
        score = min(score, 2)
    if low_energy_count >= 4:
        score = min(score, 3)

    reasoning = (
        f"Pitch range is {range_hz} Hz (average {pitch.get('average_hz')} Hz), with {len(monotone_sections)} monotone "
        f"section(s) ({'high' if high_severity_monotone else 'no high-severity'} severity) and {low_energy_count} "
        f"low-energy section(s) flagged."
    )
    return score, reasoning


def _score_engagement(fluency: Dict[str, Any], intonation: Dict[str, Any], rhythm: Dict[str, Any]) -> Tuple[int, str]:
    # engagement's OWN section carries no independent evidence in this
    # dataset (score/level both null -- see backend/feedback/coach_packets.py's
    # "executive-summary seed ONLY" rule; scores are the teacher's job, not
    # the pipeline's, per [[project_llm_assigns_scores_not_pipeline]]).
    #
    # The previous version treated this as "no engagement-specific evidence
    # exists" and averaged wpm/filler-rate/rhythm/pitch instead -- none of
    # which the engagement blueprint dimension actually drives (disengaged
    # vs. engaging sessions were statistically indistinguishable on those
    # four). low_energy sections in intonation.flat_sections DO carry
    # engagement-specific signal: repair_packets.py drives their count and
    # per-section severity directly off the engagement blueprint level
    # (disengaged: 2-4 sections skewed high-severity; engaging: 0-1, rarely
    # high). That's the primary metric here. Note `duration` never reaches
    # this packet (build_coach_packets() takes level2 only), so section
    # COUNT/severity -- not a coverage fraction -- is what's actually
    # visible; the fraction-of-duration judgment is baked into severity at
    # repair time, while duration is still available there.
    flat_sections = intonation.get("flat_sections", [])
    low_energy_sections = [s for s in flat_sections if s.get("kind") == "low_energy"]
    low_energy_count = len(low_energy_sections)
    high_severity_count = sum(1 for s in low_energy_sections if s.get("severity") == "high")

    wpm = fluency.get("speaking_speed", {}).get("words_per_minute") or 0
    pauses = fluency.get("pauses", {})
    pause_burden = (
        2.0 * pauses.get("dead_air_count", 0)
        + 1.5 * pauses.get("long_pause_count", 0)
        + 1.0 * pauses.get("hesitation_count", 0)
    )
    rhythm_pct = rhythm.get("score_pct") or 0

    # Primary: low-energy section count, a count-based staircase (same
    # shape as _score_mti's total-pattern-count staircase).
    if low_energy_count == 0:
        score = 5
    elif low_energy_count == 1:
        score = 4
    elif low_energy_count == 2:
        score = 3
    elif low_energy_count == 3:
        score = 2
    else:
        score = 1

    # Cap (never subtract) by secondary signals -- mirrors
    # _score_pronunciation/_score_mti/_score_intonation's fix.
    if high_severity_count >= 2:
        score = min(score, 2)
    elif high_severity_count == 1:
        score = min(score, 3)
    if pause_burden >= 6:
        score = min(score, 2)
    elif pause_burden >= 4:
        score = min(score, 3)
    elif pause_burden >= 2.5:
        score = min(score, 4)
    # Both a metronomic (>95%) AND an erratic (<40%) rhythm read as
    # disengaged -- rhythm_score is NOT monotonic with engagement, unlike
    # the old formula which treated "more uniform" as "more engaging."
    if rhythm_pct > 95 or rhythm_pct < 40:
        score = min(score, 4)
    if not (70 <= wpm <= 200):
        score = min(score, 4)

    reasoning = (
        f"{low_energy_count} low-energy section(s) detected ({high_severity_count} high-severity), the primary "
        f"engagement signal available in this packet, with a pause burden of {pause_burden:.1f} "
        f"({pauses.get('dead_air_count', 0)} dead-air, {pauses.get('long_pause_count', 0)} long, "
        f"{pauses.get('hesitation_count', 0)} hesitation), {rhythm_pct}% rhythm, and {wpm} wpm as secondary caps."
    )
    return score, reasoning


# ---------------------------------------------------------------------------
# Delivery: coach_output + reasoning_trace assembly
# ---------------------------------------------------------------------------
def _build_delivery_coach_output(
    session_id: str,
    fluency: Dict[str, Any],
    intonation: Dict[str, Any],
    rhythm: Dict[str, Any],
    fluency_score: int,
    intonation_score: int,
    engagement_score: int,
) -> Dict[str, Any]:
    wpm = fluency.get("speaking_speed", {}).get("words_per_minute") or 0
    filler_count = fluency.get("filler_usage", {}).get("count", 0)
    filler_per_min = fluency.get("filler_usage", {}).get("per_minute", 0.0)
    pauses = fluency.get("pauses", {})
    pitch = intonation.get("pitch", {})
    range_hz = pitch.get("range_hz") or 0
    flat_sections = sorted(intonation.get("flat_sections", []), key=lambda s: severity_rank(s.get("severity", "")), reverse=True)
    rhythm_pct = rhythm.get("score_pct") or 0

    overall_score = round((fluency_score + intonation_score + engagement_score) / 3)
    level = _level_for_score(overall_score)
    pace_note = "within a comfortable range" if 110 <= wpm <= 160 else ("above a comfortable pace" if wpm > 160 else "below a comfortable pace")
    summary = (
        f"Speaking pace is {wpm} wpm, {pace_note}. Pitch range is {range_hz} Hz and rhythm scored {rhythm_pct}%, "
        f"placing overall delivery at a {level.replace('_', ' ')} level."
    )

    strengths: List[Dict[str, str]] = []
    if filler_count == 0:
        strengths.append({"title": "No filler words", "description": "The candidate used no fillers, maintaining clarity throughout the response."})
    if range_hz >= 70:
        strengths.append({"title": "Expressive pitch", "description": f"Pitch range of {range_hz} Hz reflects clear intonational variety."})
    if rhythm_pct >= 70:
        strengths.append({"title": "Regular rhythm", "description": f"Rhythm scored {rhythm_pct}%, indicating consistent timing across the response."})
    if not strengths:
        strengths.append({"title": "Response completed", "description": "The candidate produced a complete, analyzable response despite the delivery issues below."})

    perceived_confidence = "high" if overall_score >= 4 else ("moderate" if overall_score == 3 else "low")
    communication_flow = "smooth" if pauses.get("dead_air_count", 0) == 0 and pauses.get("long_pause_count", 0) == 0 else "disrupted by pauses"
    interviewer_impression = {
        "perceived_confidence": perceived_confidence,
        "perceived_engagement": _level_for_score(engagement_score).replace("_", " "),
        "communication_flow": communication_flow,
        "professional_presence": "developing" if overall_score <= 2 else "adequate",
        "overall_impact": f"Overall delivery reads as {level.replace('_', ' ')} based on pace, pitch, and rhythm evidence.",
    }

    priority_improvements: List[Dict[str, Any]] = []
    priority = 1
    if not (110 <= wpm <= 160):
        priority_improvements.append({
            "priority": priority,
            "issue": "Speaking pace",
            "impact": "high" if wpm > 190 or wpm < 80 else "medium",
            "why_it_matters": f"{wpm} wpm falls outside the 120-150 wpm comfortable range for interview responses.",
        })
        priority += 1
    if filler_per_min >= 5:
        priority_improvements.append({
            "priority": priority,
            "issue": "Filler word usage",
            "impact": "high" if filler_per_min >= 10 else "medium",
            "why_it_matters": f"{filler_count} filler(s) at {filler_per_min}/min interrupts otherwise fluent delivery.",
        })
        priority += 1
    if range_hz < 45:
        priority_improvements.append({
            "priority": priority,
            "issue": "Pitch variation",
            "impact": "medium",
            "why_it_matters": f"A {range_hz} Hz pitch range risks reading as flat or monotone to a listener.",
        })
        priority += 1
    if not priority_improvements:
        priority_improvements.append({
            "priority": 1,
            "issue": "Maintain current delivery",
            "impact": "low",
            "why_it_matters": "No delivery metric fell outside its comfortable range in this response.",
        })

    detailed_findings = [{
        "category": "pace",
        "severity": "high" if not (90 <= wpm <= 180) else "low",
        "observation": f"Sustained {wpm} wpm across the response.",
        "impact": "Listener comprehension drops sharply above ~180 wpm or below ~90 wpm." if not (90 <= wpm <= 180) else "Pace is comfortable for listener comprehension.",
        "evidence": {"words_per_minute": wpm, "classification": pace_note},
    }]
    if filler_count > 0:
        detailed_findings.append({
            "category": "fillers",
            "severity": "high" if filler_per_min >= 10 else ("medium" if filler_per_min >= 5 else "low"),
            "observation": f"{filler_count} filled pause(s) ({filler_per_min}/min).",
            "impact": "Signals hesitation mid-thought.",
            "evidence": {"examples": fluency.get("filler_usage", {}).get("examples", [])[:5]},
        })
    if flat_sections:
        detailed_findings.append({
            "category": "energy" if flat_sections[0].get("kind") == "low_energy" else "rhythm",
            "severity": flat_sections[0].get("severity", "medium"),
            "observation": f"{len(flat_sections)} flat/low-energy section(s) detected in pitch or loudness.",
            "impact": "Flat stretches read as disengaged to a listener.",
            "evidence": {"timestamps": [{"start": s.get("start"), "end": s.get("end")} for s in flat_sections[:5]]},
        })
    detailed_findings.append({
        "category": "pauses",
        "severity": "medium" if pauses.get("dead_air_count", 0) or pauses.get("long_pause_count", 0) else "low",
        "observation": f"{pauses.get('total', 0)} pause(s), {pauses.get('hesitation_count', 0)} hesitation(s), {pauses.get('dead_air_count', 0)} dead air.",
        "impact": "Pause structure is not a problem." if not pauses.get("dead_air_count") else "Dead air interrupts the flow of the response.",
        "evidence": {"dead_air": pauses.get("dead_air_count", 0), "long_pauses": pauses.get("long_pause_count", 0), "hesitations": pauses.get("hesitation_count", 0)},
    })

    timeline_review = [
        {"time": f"{_fmt_time(s.get('start'))}-{_fmt_time(s.get('end'))}", "event": f"{s.get('kind', 'flat').replace('_', ' ').title()} section"}
        for s in flat_sections[:3]
    ]

    behavioral_patterns = []
    if wpm > 160 and filler_per_min > 0:
        behavioral_patterns.append({"pattern": "Rushing under pressure", "frequency": "throughout", "effect": "Compresses content and invites filled pauses."})
    elif wpm < 100:
        behavioral_patterns.append({"pattern": "Deliberate, cautious pacing", "frequency": "throughout", "effect": "May read as under-confident despite accurate content."})
    if not behavioral_patterns:
        behavioral_patterns.append({"pattern": "Steady delivery", "frequency": "throughout", "effect": "No dominant behavioral pattern stood out in this response."})

    practice_plan = []
    if not (110 <= wpm <= 160):
        practice_plan.append({
            "focus": "Pace control",
            "exercise": "Read aloud to a metronome at 140 wpm for 2 minutes, then answer a question at that pace.",
        })
    if filler_per_min >= 5:
        practice_plan.append({"focus": "Silent pausing", "exercise": "Replace each filler with a closed-mouth beat of silence."})
    if range_hz < 45:
        practice_plan.append({"focus": "Pitch variety", "exercise": "Read a paragraph aloud exaggerating pitch rises on key words, then dial it back to natural."})
    if not practice_plan:
        practice_plan.append({"focus": "Maintain current delivery", "exercise": "Continue practicing under time pressure to keep these metrics stable."})

    coach_priority = {
        "fix_first": priority_improvements[0]["issue"],
        "reason": priority_improvements[0]["why_it_matters"],
        "estimated_delivery_gain": "+10-20 points" if priority_improvements[0]["impact"] == "high" else "+5-10 points",
    }

    next_focus = f"Hold {max(110, min(160, wpm))}-150 wpm for a full answer." if not (110 <= wpm <= 160) else "Maintain current pace and pitch variety in the next session."

    return {
        "session_id": session_id,
        "coach": "delivery",
        "overall_assessment": {"delivery_level": level, "summary": summary},
        "interviewer_impression": interviewer_impression,
        "strengths": strengths,
        "priority_improvements": priority_improvements,
        "detailed_findings": detailed_findings,
        "timeline_review": timeline_review,
        "behavioral_patterns": behavioral_patterns,
        "practice_plan": practice_plan,
        "coach_priority": coach_priority,
        "next_session_focus": [next_focus],
    }


def _build_delivery_reasoning_trace(
    fluency_score: int, fluency_reasoning: str,
    intonation_score: int, intonation_reasoning: str,
    engagement_score: int, engagement_reasoning: str,
    coach_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    trace = [
        {"conclusion": "fluency rubric score", "evidence_cited": ["fluency.speaking_speed", "fluency.filler_usage", "fluency.pauses"], "reasoning": fluency_reasoning},
        {"conclusion": "intonation rubric score", "evidence_cited": ["intonation.pitch", "intonation.flat_sections"], "reasoning": intonation_reasoning},
        {"conclusion": "engagement rubric score", "evidence_cited": ["fluency", "intonation", "rhythm"], "reasoning": engagement_reasoning},
    ]
    for imp in coach_output["priority_improvements"]:
        trace.append({
            "conclusion": f"priority_improvements: {imp['issue']}",
            "evidence_cited": [imp["why_it_matters"]],
            "reasoning": f"Ranked priority {imp['priority']} because it carries {imp['impact']} impact per the evidence cited.",
        })
    trace.append({
        "conclusion": f"coach_priority: {coach_output['coach_priority']['fix_first']}",
        "evidence_cited": [coach_output["coach_priority"]["reason"]],
        "reasoning": "Selected as the single highest-leverage fix among this session's priority_improvements.",
    })
    return trace


def synthesize_delivery(evidence: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Produce the {scores, coach_output, reasoning_trace} document a
    Delivery teacher (MiMo-v2.5, or here, Claude) would return for this
    evidence packet."""
    fluency = evidence.get("fluency", {})
    intonation = evidence.get("intonation", {})
    rhythm = evidence.get("rhythm", {})

    fluency_score, fluency_reasoning = _score_fluency(fluency)
    intonation_score, intonation_reasoning = _score_intonation(intonation)
    engagement_score, engagement_reasoning = _score_engagement(fluency, intonation, rhythm)

    coach_output = _build_delivery_coach_output(
        session_id, fluency, intonation, rhythm, fluency_score, intonation_score, engagement_score
    )
    reasoning_trace = _build_delivery_reasoning_trace(
        fluency_score, fluency_reasoning, intonation_score, intonation_reasoning,
        engagement_score, engagement_reasoning, coach_output,
    )

    return {
        "scores": {
            "fluency": {"score": fluency_score, "reasoning": fluency_reasoning},
            "intonation": {"score": intonation_score, "reasoning": intonation_reasoning},
            "engagement": {"score": engagement_score, "reasoning": engagement_reasoning},
        },
        "coach_output": coach_output,
        "reasoning_trace": reasoning_trace,
    }


class ClaudeTeacherProvider:
    """Drop-in replacement for TeacherProvider (provider.py) -- same two
    methods, same signatures, same return type (a raw JSON string). Use it
    with TeacherRunner exactly like the real provider:

        TeacherRunner(provider=ClaudeTeacherProvider())

    Construction never touches the network (there is none to touch) --
    this class is entirely local computation.
    """

    name = "claude"

    def generate_articulation(self, prompt: str) -> str:
        session_id, evidence = _extract_session_and_evidence(prompt)
        result = synthesize_articulation(evidence, session_id)
        return json.dumps(result, indent=2, ensure_ascii=False)

    def generate_delivery(self, prompt: str) -> str:
        session_id, evidence = _extract_session_and_evidence(prompt)
        result = synthesize_delivery(evidence, session_id)
        return json.dumps(result, indent=2, ensure_ascii=False)
