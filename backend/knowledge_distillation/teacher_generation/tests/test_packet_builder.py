"""
Tests for teacher_generation.packet_builder. All local/pure -- calls the
real build_coach_packets() (no network) against hand-built evidence dicts.
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

from backend.knowledge_distillation.teacher_generation import (
    CoachPacketPair,
    PacketBuildError,
    build_all,
    build_packet_pair,
    iter_evidence_packets,
    write_packet_pairs,
)

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


def make_evidence_packet(session_id="session_test_1", level2=None):
    return {
        "session_id": session_id,
        "blueprint": {"blueprint_id": "bp_1", "pronunciation": "strong"},
        "level1": {"audio_metadata": {"duration": 5.0}},
        "level2": level2 if level2 is not None else GOOD_LEVEL2,
        "metadata": {"model": "claude-code-session"},
    }


class BuildPacketPairTests(unittest.TestCase):
    def test_returns_a_typed_pair_with_real_coach_packets(self):
        pair = build_packet_pair(make_evidence_packet())
        self.assertIsInstance(pair, CoachPacketPair)
        self.assertEqual(pair.session_id, "session_test_1")
        self.assertEqual(pair.articulation["coach"], "articulation")
        self.assertEqual(pair.delivery["coach"], "delivery")
        # score fields are None -- no scores exist in the input (see
        # synthetic_generation's config.py docstring); this is expected.
        self.assertIsNone(pair.articulation["pronunciation"]["score"])

    def test_missing_session_id_raises(self):
        bad = make_evidence_packet()
        del bad["session_id"]
        with self.assertRaises(PacketBuildError):
            build_packet_pair(bad)

    def test_missing_level2_raises(self):
        bad = make_evidence_packet()
        del bad["level2"]
        with self.assertRaises(PacketBuildError):
            build_packet_pair(bad)

    def test_malformed_level2_is_wrapped_as_packet_build_error(self):
        # mispronounced_words entry missing "word" -- build_articulation_packet()
        # reads w["word"] via hard indexing and raises.
        level2 = {**GOOD_LEVEL2, "pronunciation": {**GOOD_LEVEL2["pronunciation"], "mispronounced_words": [{"start": 0.0, "end": 0.5}]}}
        with self.assertRaises(PacketBuildError):
            build_packet_pair(make_evidence_packet(level2=level2))


class IterEvidencePacketsTests(unittest.TestCase):
    def test_reads_one_dict_per_line(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "packets.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(make_evidence_packet("s1")) + "\n")
                f.write(json.dumps(make_evidence_packet("s2")) + "\n")
            packets = list(iter_evidence_packets(path))
            self.assertEqual([p["session_id"] for p in packets], ["s1", "s2"])

    def test_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "packets.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(make_evidence_packet("s1")) + "\n\n")
            packets = list(iter_evidence_packets(path))
            self.assertEqual(len(packets), 1)


class BuildAllTests(unittest.TestCase):
    def test_streams_a_pair_per_packet(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "packets.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(5):
                    f.write(json.dumps(make_evidence_packet(f"s{i}")) + "\n")
            pairs = list(build_all(path))
            self.assertEqual(len(pairs), 5)
            self.assertEqual([p.session_id for p in pairs], [f"s{i}" for i in range(5)])


class WritePacketPairsTests(unittest.TestCase):
    def test_writes_articulation_and_delivery_files(self):
        with tempfile.TemporaryDirectory() as d:
            packets_path = os.path.join(d, "packets.jsonl")
            with open(packets_path, "w", encoding="utf-8") as f:
                for i in range(6):
                    f.write(json.dumps(make_evidence_packet(f"s{i}")) + "\n")

            out_dir = os.path.join(d, "out")
            count = write_packet_pairs(packets_path, out_dir)
            self.assertEqual(count, 6)

            art_path = os.path.join(out_dir, "articulation", "articulation.jsonl")
            del_path = os.path.join(out_dir, "delivery", "delivery.jsonl")
            with open(art_path) as f:
                art_lines = f.readlines()
            with open(del_path) as f:
                del_lines = f.readlines()
            self.assertEqual(len(art_lines), 6)
            self.assertEqual(len(del_lines), 6)

            first = json.loads(art_lines[0])
            self.assertEqual(set(first.keys()), {"session_id", "coach", "pronunciation", "mti"})
            first_delivery = json.loads(del_lines[0])
            self.assertEqual(set(first_delivery.keys()), {"session_id", "coach", "fluency", "intonation", "rhythm", "engagement"})


class ModuleBoundaryTests(unittest.TestCase):
    def test_packet_builder_never_calls_a_teacher_model(self):
        """packet_builder.py derives coach-packet INPUTS only -- it must
        never import a provider/LLM client, which would mean it's trying to
        call a teacher model itself instead of leaving that to provider.py
        (Part 2) and a later teacher-execution stage. Scoped to this one
        file, not the whole package: provider.py legitimately imports
        llm.client -- that IS its job, see its own module docstring."""
        import ast

        forbidden = ("anthropic", "llm.client", "llm.anthropic_client")
        path = os.path.join(_PACKAGE_DIR, "packet_builder.py")
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        offending = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(phrase in name.lower() for phrase in forbidden):
                    offending.append(name)
        self.assertEqual(offending, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
