"""
Speaking dynamics analysis for engagement: combines pitch variation (intonation),
rhythm (pronunciation), speaking speed (fluency), and emphasis (intonation) to
judge whether delivery reads as animated/expressive or robotic/flat.

No new audio analysis happens here — reuses scores and observations already
computed by fluency, pronunciation, and intonation.

The strengths/issues here are written to never contradict each other: vocal
variation gets a single verdict (one strength OR one issue, never "good pitch
variation" alongside "sounds monotone"), and when delivery is monotone the
emphasis complaint is folded into that one line instead of listed separately.
"""

from typing import Dict, List, Tuple

# Vocal variation reads as "good" only when BOTH the whole-recording pitch range
# is healthy AND there are no sustained flat stretches. One threshold failing is
# enough to make it an issue — that's what stops the score from disagreeing with
# the monotone finding.
PITCH_OK_THRESHOLD = 70
MONOTONICITY_OK_THRESHOLD = 60

# Fix 7 (audio-pipeline correctness-fix plan): pace previously only affected the
# TEXT (issues/strengths) via _speed_strength_or_issue, never the numeric
# dynamics_score — so a recording could be scored "highly engaging" while
# separately listing "Speaking pace is too fast" as an issue, a direct
# contradiction between the number and the text describing it. Giving pace a
# real numeric weight (same as the other four components) means a bad pace
# classification now actually pulls the score down, not just the prose.
_PACE_SCORE = {
    "ideal": 100,
    "fast": 75,
    "slow": 75,
    "too_fast": 50,
    "very_slow": 50,
}
_DEFAULT_PACE_SCORE = 75  # unrecognized classification: neutral, not perfect


def _pace_score(classification: str) -> int:
    return _PACE_SCORE.get(classification, _DEFAULT_PACE_SCORE)


def _classify_delivery_character(average_score: float) -> str:
    if average_score >= 80:
        return "animated and expressive"
    if average_score >= 60:
        return "moderately expressive"
    if average_score >= 40:
        return "somewhat flat"
    return "robotic and flat"


def _speed_strength_or_issue(classification: str) -> Tuple[List[str], List[str]]:
    strengths, issues = [], []
    if classification == "ideal":
        strengths.append("Comfortable speaking pace")
    elif classification in ("too_fast", "very_slow"):
        issues.append(f"Speaking pace is {classification.replace('_', ' ')}")
    elif classification in ("fast", "slow"):
        issues.append(f"Speaking pace runs slightly {classification}")
    return strengths, issues


def analyze_speaking_dynamics(
    fluency_output: Dict, pronunciation_output: Dict, intonation_output: Dict
) -> Dict:
    """
    Judge whether delivery sounds animated/expressive or robotic/flat, combining
    pitch variation + emphasis (intonation), rhythm (pronunciation), and
    speaking speed (fluency).

    Input: the full dicts returned by fluency_analyzer.analyze_fluency(),
        pronunciation_analyzer.analyze_pronunciation(), and
        intonation_analyzer.analyze_intonation().

    Output:
        {
            "dynamics_score": int,
            "strengths": [str, ...],
            "issues": [str, ...],
            "observations": [str, ...],
        }
    """
    pitch_score = intonation_output.get("pitch_variation", {}).get("pitch_score", 0)
    emphasis_score = intonation_output.get("emphasis", {}).get("emphasis_score", 0)
    monotonicity_score = intonation_output.get("monotonicity", {}).get("monotonicity_score", 0)
    rhythm_score = round(pronunciation_output.get("rhythm_score", 0) * 100)
    speed_classification = fluency_output.get("speaking_speed", {}).get("classification", "")

    # monotonicity_score is a component, not just a flag: it's a direct measure of
    # animated-vs-flat delivery (the exact question this file answers), so leaving
    # it out would let a near-flat recording still score as highly dynamic while
    # the timeline (which uses monotone_sections) says otherwise. pace_score is
    # the same discipline applied to speed (Fix 7) — previously only affected
    # issues/strengths text, never this number.
    pace_score = _pace_score(speed_classification)
    component_scores = [pitch_score, rhythm_score, emphasis_score, monotonicity_score, pace_score]
    average_score = sum(component_scores) / len(component_scores)
    dynamics_score = round(average_score)

    strengths: List[str] = []
    issues: List[str] = []

    # --- Single vocal-variation verdict (pitch + monotonicity + emphasis) -------
    vocal_variation_good = (
        pitch_score >= PITCH_OK_THRESHOLD and monotonicity_score >= MONOTONICITY_OK_THRESHOLD
    )
    emphasis_weak = emphasis_score < 70
    if vocal_variation_good:
        strengths.append("Good vocal variation — pitch moves naturally and delivery stays lively")
        if emphasis_weak:
            # Variation is fine overall but key words still aren't landing — a
            # narrower, non-contradictory note.
            issues.append("Key words could stand out more with a little extra emphasis")
    else:
        # Flat delivery: state it once, and fold the emphasis complaint into the
        # same line rather than repeating it as a separate issue.
        if emphasis_weak:
            issues.append("Delivery sounds flat — pitch stays level and key words don't stand out")
        else:
            issues.append("Delivery sounds flat — limited pitch movement across the answer")

    # --- Pace (independent dimension) ------------------------------------------
    speed_strengths, speed_issues = _speed_strength_or_issue(speed_classification)
    strengths.extend(speed_strengths)
    issues.extend(speed_issues)

    # --- Rhythm (timing regularity, independent of pitch flatness) -------------
    if rhythm_score < 60:
        issues.append("Delivery rhythm feels irregular")

    character = _classify_delivery_character(average_score)
    observations = [f"Speech delivery reads as {character}."]

    return {
        "dynamics_score": dynamics_score,
        "strengths": strengths,
        "issues": issues,
        "observations": observations,
    }
