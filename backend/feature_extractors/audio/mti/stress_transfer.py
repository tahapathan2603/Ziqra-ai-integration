"""
Stress transfer detection for MTI analysis: stress placement carried over from
native-language speech rhythm rather than the target word's actual stress pattern.

PLACEHOLDER-DEPENDENT: this reuses pronunciation/stress_placement.py's stress
comparison, which is itself a placeholder — it has no real acoustic stress
detection yet (see the PLACEHOLDER comment in stress_placement.py's
_detect_stress()), so detected stress currently always equals expected stress.
That means stress_errors will always be empty and this file will always report
no stress transfer issues, until real pitch/energy-based stress detection
(e.g. librosa.pyin or CREPE) is integrated there. This file's own logic
(severity assignment, timestamp attachment, output shape) is real and ready to
use real data the moment stress_placement.py's detection is.
"""

from typing import Dict, List

from .phoneme_substitution import build_word_timestamps

# Placeholder severity for any stress transfer issue, since stress_placement.py
# doesn't yet produce a severity signal of its own (no real acoustic detection to
# grade). Revisit once real stress detection lands.
_DEFAULT_SEVERITY = "medium"


def analyze_stress_transfer(stress_errors: List[Dict], words: List[Dict]) -> Dict:
    """
    Attach timestamps to stress placement errors for MTI reporting.

    Input:
        stress_errors: [{"word": str, "expected": str, "detected": str}, ...]
            (the "stress_errors" list from stress_placement.analyze_stress();
            "expected"/"detected" are rendered strings like "preSENT"/"PREsent")
        words: [{"word": str, "start": float, "end": float}, ...]

    Output:
        {
            "stress_transfer": [
                {
                    "word": str,
                    "timestamp": {"start": float, "end": float},
                    "expected_stress": str,
                    "detected_stress": str,
                    "severity": str,
                },
                ...
            ]
        }
    """
    timestamps = build_word_timestamps(words)
    transfers = []

    for err in stress_errors:
        ts = timestamps.get(err["word"], {"start": None, "end": None})
        transfers.append(
            {
                "word": err["word"],
                "timestamp": {"start": ts["start"], "end": ts["end"]},
                "expected_stress": err["expected"],
                "detected_stress": err["detected"],
                "severity": _DEFAULT_SEVERITY,
            }
        )

    return {"stress_transfer": transfers}
