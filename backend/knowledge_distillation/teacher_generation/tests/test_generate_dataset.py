"""
Tests for teacher_generation.generate_dataset. Uses a fake provider (no
network, no credentials) and a tiny synthetic packets.jsonl written to a
temp dir.
"""

import json
import os
import sys
import tempfile
import unittest

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_MODULE_DIR)
_KD_DIR = os.path.dirname(_PACKAGE_DIR)
_BACKEND_DIR = os.path.dirname(_KD_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from backend.knowledge_distillation.teacher_generation.exceptions import NonRetryableTeacherError, TeacherProviderError
from backend.knowledge_distillation.teacher_generation.generate_dataset import run

GOOD_LEVEL2 = {
    "fluency": {
        "fillers": {"filler_count": 0, "fillers_per_minute": 0.0, "fillers": []},
        "pauses": {"total_pauses": 0, "pauses": []},
        "speaking_speed": {"words_per_minute": 120, "sentences_per_minute": 10},
    },
    "pronunciation": {
        "phoneme_accuracy": 0.9, "stress_accuracy": 1.0, "rhythm_score": 0.9,
        "phoneme_errors": [], "detected_phonemes": [],
        "mispronounced_words": [{"word": "hello", "severity": "low", "start": 0.0, "end": 0.5}],
        "stress_errors": [], "rhythm_issues": [],
    },
    "mti": {
        "summary": {"vowel_pattern_issues": 0, "consonant_pattern_issues": 0, "stress_transfer_issues": 0},
        "vowel_patterns": [], "consonant_patterns": [], "stress_transfer": [],
        "speech_statistics": {"total_words_analyzed": 5, "affected_words": 0, "affected_percentage": 0.0},
    },
    "intonation": {
        "pitch_variation": {"average_pitch": 150.0, "min_pitch": 120.0, "max_pitch": 180.0, "pitch_range": 60.0},
        "energy_variation": {"low_energy_sections": []},
        "monotonicity": {"monotone_sections": []},
        "emphasis": {"under_emphasized_words": []},
    },
    "engagement": {},
}


def _write_packets(path, session_ids):
    with open(path, "w", encoding="utf-8") as f:
        for sid in session_ids:
            f.write(json.dumps({"session_id": sid, "level2": GOOD_LEVEL2}) + "\n")


class FakeProvider:
    name = "fake"

    def __init__(self, fail_on=None, non_retryable_on=None):
        self._fail_on = fail_on or set()
        self._non_retryable_on = non_retryable_on or set()
        self.articulation_calls = []
        self.delivery_calls = []

    def generate_articulation(self, prompt):
        self.articulation_calls.append(prompt)
        return "articulation raw"

    def generate_delivery(self, prompt):
        self.delivery_calls.append(prompt)
        return "delivery raw"


class RunTests(unittest.TestCase):
    def test_writes_both_files_tagged_with_provider_name(self):
        with tempfile.TemporaryDirectory() as d:
            packets_path = os.path.join(d, "packets.jsonl")
            _write_packets(packets_path, ["s1", "s2"])
            out_dir = os.path.join(d, "out")

            run(packets_path=packets_path, out_dir=out_dir, provider=FakeProvider())

            with open(os.path.join(out_dir, "articulation_raw.jsonl")) as f:
                art_lines = [json.loads(l) for l in f]
            with open(os.path.join(out_dir, "delivery_raw.jsonl")) as f:
                del_lines = [json.loads(l) for l in f]

            self.assertEqual([r["session_id"] for r in art_lines], ["s1", "s2"])
            self.assertEqual([r["generated_by"] for r in art_lines], ["fake", "fake"])
            self.assertEqual([r["raw_response"] for r in art_lines], ["articulation raw", "articulation raw"])
            self.assertEqual([r["session_id"] for r in del_lines], ["s1", "s2"])

    def test_resume_skips_already_done_sessions(self):
        with tempfile.TemporaryDirectory() as d:
            packets_path = os.path.join(d, "packets.jsonl")
            _write_packets(packets_path, ["s1", "s2"])
            out_dir = os.path.join(d, "out")

            provider1 = FakeProvider()
            run(packets_path=packets_path, out_dir=out_dir, provider=provider1)
            self.assertEqual(len(provider1.articulation_calls), 2)

            provider2 = FakeProvider()
            run(packets_path=packets_path, out_dir=out_dir, provider=provider2)
            self.assertEqual(len(provider2.articulation_calls), 0)  # both sessions already done

    def test_respects_limit(self):
        with tempfile.TemporaryDirectory() as d:
            packets_path = os.path.join(d, "packets.jsonl")
            _write_packets(packets_path, ["s1", "s2", "s3"])
            out_dir = os.path.join(d, "out")

            run(packets_path=packets_path, out_dir=out_dir, provider=FakeProvider(), limit=1)

            with open(os.path.join(out_dir, "articulation_raw.jsonl")) as f:
                self.assertEqual(len(f.readlines()), 1)

    def test_ordinary_provider_error_is_skipped_not_fatal(self):
        class FlakyProvider(FakeProvider):
            def generate_articulation(self, prompt):
                raise TeacherProviderError("exhausted retries")

        with tempfile.TemporaryDirectory() as d:
            packets_path = os.path.join(d, "packets.jsonl")
            _write_packets(packets_path, ["s1", "s2"])
            out_dir = os.path.join(d, "out")

            run(packets_path=packets_path, out_dir=out_dir, provider=FlakyProvider())

            # articulation failed for both -> file has 0 lines; delivery succeeded -> 2 lines
            with open(os.path.join(out_dir, "articulation_raw.jsonl")) as f:
                self.assertEqual(f.readlines(), [])
            with open(os.path.join(out_dir, "delivery_raw.jsonl")) as f:
                self.assertEqual(len(f.readlines()), 2)

    def test_non_retryable_error_aborts_the_whole_run(self):
        class BrokenProvider(FakeProvider):
            def generate_articulation(self, prompt):
                raise NonRetryableTeacherError("no credits")

        with tempfile.TemporaryDirectory() as d:
            packets_path = os.path.join(d, "packets.jsonl")
            _write_packets(packets_path, ["s1", "s2"])
            out_dir = os.path.join(d, "out")

            with self.assertRaises(NonRetryableTeacherError):
                run(packets_path=packets_path, out_dir=out_dir, provider=BrokenProvider())

            # aborted on s1 -- delivery for s1 never got a chance either
            with open(os.path.join(out_dir, "delivery_raw.jsonl")) as f:
                self.assertEqual(f.readlines(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
