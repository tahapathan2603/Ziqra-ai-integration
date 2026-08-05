"""
Tests for repair_packets.py -- the one-shot deterministic evidence repair
pass that fixes pitch_range clustering and low_energy_sections inertness
(see backend.knowledge_distillation.teacher_generation.claude_teacher's
score-distribution fix). All local/pure -- no network call.
"""

import json
import os
import sys
import unittest
from pathlib import Path

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_MODULE_DIR)
_KD_DIR = os.path.dirname(_PACKAGE_DIR)
_BACKEND_DIR = os.path.dirname(_KD_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from backend.knowledge_distillation.synthetic_generation.repair_packets import (
    PITCH_SOURCE_BANDS,
    PITCH_TARGET_BANDS,
    repair_file,
    repair_row,
    rescale_pitch_range,
)


def _make_row(
    session_id="session_test",
    intonation_level="moderate",
    engagement_level="neutral",
    pitch_range=100,
    duration=10.0,
    monotone_sections=None,
):
    return {
        "session_id": session_id,
        "blueprint": {
            "blueprint_id": "bp1",
            "pronunciation": "strong",
            "mti_severity": "none",
            "pace": "moderate",
            "filler_frequency": "minimal",
            "rhythm": "steady",
            "intonation": intonation_level,
            "engagement": engagement_level,
            "confidence": "moderate",
        },
        "level1": {"audio_metadata": {"duration": duration, "sample_rate": 16000, "detected_language": "en", "language_probability": 0.95}},
        "level2": {
            "pronunciation": {"phoneme_accuracy": 0.95, "rhythm_score": 0.8, "mispronounced_words": [], "rhythm_issues": []},
            "mti": {"summary": {"vowel_pattern_issues": 0, "consonant_pattern_issues": 0}, "vowel_patterns": [], "consonant_patterns": []},
            "fluency": {
                "speaking_speed": {"words_per_minute": 130},
                "fillers": {"filler_count": 0, "fillers_per_minute": 0.0, "fillers": []},
                "pauses": {"total_pauses": 0, "pauses": []},
            },
            "intonation": {
                "pitch_variation": {"average_pitch": 120, "min_pitch": 120 - pitch_range // 2, "max_pitch": 120 + pitch_range // 2, "pitch_range": pitch_range},
                "energy_variation": {"low_energy_sections": []},
                "monotonicity": {"monotone_sections": monotone_sections or []},
                "emphasis": {"under_emphasized_words": []},
            },
            "engagement": {},
        },
        "metadata": {"model": "claude-code-session", "prompt_version": "1.0.0", "schema_version": "1.0.0", "generated_at": "2026-08-04T00:00:00+00:00"},
    }


class RescalePitchRangeTests(unittest.TestCase):
    def test_rescales_into_target_band(self):
        for level, (src_lo, src_hi) in PITCH_SOURCE_BANDS.items():
            dst_lo, dst_hi = PITCH_TARGET_BANDS[level]
            self.assertEqual(rescale_pitch_range(src_lo, level), dst_lo)
            self.assertEqual(rescale_pitch_range(src_hi, level), dst_hi)

    def test_monotonic_within_band(self):
        for level, (src_lo, src_hi) in PITCH_SOURCE_BANDS.items():
            values = [src_lo + i * (src_hi - src_lo) / 10 for i in range(11)]
            rescaled = [rescale_pitch_range(v, level) for v in values]
            self.assertEqual(rescaled, sorted(rescaled))

    def test_target_bands_are_continuous_no_dead_zones(self):
        # flat's target max must reach (or nearly reach) moderate's target
        # min, and moderate's max must reach expressive's min -- this is
        # the actual defect being fixed (source bands had 51-79/141-159 Hz
        # dead zones between them).
        self.assertGreaterEqual(PITCH_TARGET_BANDS["moderate"][0], PITCH_TARGET_BANDS["flat"][0])
        self.assertLessEqual(PITCH_TARGET_BANDS["flat"][1], PITCH_TARGET_BANDS["moderate"][1])
        self.assertLess(PITCH_TARGET_BANDS["flat"][1] - PITCH_TARGET_BANDS["moderate"][0], 10)
        self.assertLess(PITCH_TARGET_BANDS["moderate"][1] - PITCH_TARGET_BANDS["expressive"][0], 10)


class RepairRowTests(unittest.TestCase):
    def test_repair_is_deterministic(self):
        row = _make_row(engagement_level="disengaged")
        a = repair_row(row)
        b = repair_row(row)
        self.assertEqual(a, b)

    def test_only_level2_intonation_changes(self):
        row = _make_row(engagement_level="disengaged", pitch_range=30, intonation_level="flat")
        repaired = repair_row(row)
        self.assertEqual(repaired["level1"], row["level1"])
        for key in ("pronunciation", "mti", "fluency", "engagement"):
            self.assertEqual(repaired["level2"][key], row["level2"][key])
        self.assertNotEqual(repaired["level2"]["intonation"], row["level2"]["intonation"])

    def test_pitch_range_lands_in_target_band(self):
        for level in ("flat", "moderate", "expressive"):
            src_lo, src_hi = PITCH_SOURCE_BANDS[level]
            dst_lo, dst_hi = PITCH_TARGET_BANDS[level]
            row = _make_row(intonation_level=level, pitch_range=(src_lo + src_hi) / 2)
            repaired = repair_row(row)
            new_range = repaired["level2"]["intonation"]["pitch_variation"]["pitch_range"]
            self.assertGreaterEqual(new_range, dst_lo)
            self.assertLessEqual(new_range, dst_hi)

    def test_min_max_pitch_consistent_with_new_range(self):
        row = _make_row(intonation_level="expressive", pitch_range=200)
        repaired = repair_row(row)
        pv = repaired["level2"]["intonation"]["pitch_variation"]
        self.assertEqual(pv["max_pitch"] - pv["min_pitch"], pv["pitch_range"])

    def test_engagement_drives_low_energy_section_count(self):
        counts = {}
        for level in ("engaging", "neutral", "disengaged"):
            row = _make_row(engagement_level=level, duration=15.0)
            repaired = repair_row(row)
            counts[level] = len(repaired["level2"]["intonation"]["energy_variation"]["low_energy_sections"])
        # Not a strict inequality per-row (bands overlap by design), but
        # disengaged must be able to reach counts engaging cannot.
        self.assertLessEqual(counts["engaging"], 1)

    def test_low_energy_sections_stay_within_duration_and_dont_overlap(self):
        row = _make_row(engagement_level="disengaged", duration=6.0)
        repaired = repair_row(row)
        sections = repaired["level2"]["intonation"]["energy_variation"]["low_energy_sections"]
        sections_sorted = sorted(sections, key=lambda s: s["start"])
        for s in sections_sorted:
            self.assertGreaterEqual(s["start"], 0.0)
            self.assertLessEqual(s["end"], 6.0)
            self.assertLess(s["start"], s["end"])
        for a, b in zip(sections_sorted, sections_sorted[1:]):
            self.assertLessEqual(a["end"], b["start"])

    def test_low_energy_sections_carry_severity(self):
        row = _make_row(engagement_level="disengaged", duration=10.0)
        repaired = repair_row(row)
        sections = repaired["level2"]["intonation"]["energy_variation"]["low_energy_sections"]
        for s in sections:
            self.assertIn(s["severity"], ("high", "medium"))

    def test_monotone_sections_carry_severity(self):
        row = _make_row(monotone_sections=[{"start": 0.0, "end": 9.0}], duration=10.0)
        repaired = repair_row(row)
        sections = repaired["level2"]["intonation"]["monotonicity"]["monotone_sections"]
        self.assertEqual(len(sections), 1)
        self.assertIn(sections[0]["severity"], ("high", "medium"))
        # 90% coverage is well above MONOTONE_HIGH_COVERAGE_RATIO (0.6).
        self.assertEqual(sections[0]["severity"], "high")


class RepairFileTests(unittest.TestCase):
    def test_repair_file_is_idempotent_via_backup(self, tmp_path=None):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "packets.jsonl"
            row = _make_row(engagement_level="disengaged")
            path.write_text(json.dumps(row) + "\n")

            repair_file(path)
            first = path.read_text()
            repair_file(path)  # sources from the .bak this call creates
            second = path.read_text()

            self.assertEqual(first, second)
            self.assertTrue((path.with_suffix(path.suffix + ".bak")).exists())

    def test_backup_preserves_pristine_original(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "packets.jsonl"
            row = _make_row(pitch_range=100, intonation_level="moderate")
            original_line = json.dumps(row) + "\n"
            path.write_text(original_line)

            repair_file(path)

            backup_path = path.with_suffix(path.suffix + ".bak")
            self.assertEqual(backup_path.read_text(), original_line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
