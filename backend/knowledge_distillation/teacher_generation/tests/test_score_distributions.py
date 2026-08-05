"""
Calibration guard for ClaudeTeacherProvider's score distributions -- the
first distribution/coverage test in the knowledge_distillation package.
This is the test that would have caught the original defect: engagement
scored 1 for 1/2000 sessions and intonation scored 5 for 927/2000, because
_score_engagement ignored the one signal engagement blueprints actually
drove (low_energy_sections) and _score_intonation's bands didn't fit
pitch_range's evidence shape.

Runs over the real, repaired dataset
(synthetic_generation/datasets/packets/packets.jsonl, post
repair_packets.py) rather than synthetic fixtures, because the defect was
about the SHAPE of a 2000-session distribution -- a couple of hand-built
fixtures can't demonstrate that shape exists or has been fixed.

Skips (not fails) if the dataset file isn't present, e.g. in a checkout
that hasn't run the synthetic-generation step -- this test verifies a
property of the dataset, not of the code in isolation.
"""

import json
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_MODULE_DIR)
_KD_DIR = os.path.dirname(_PACKAGE_DIR)
_BACKEND_DIR = os.path.dirname(_KD_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from backend.knowledge_distillation.synthetic_generation.repair_packets import DEFAULT_PACKETS_PATH
from backend.knowledge_distillation.teacher_generation.claude_teacher import synthesize_delivery
from backend.knowledge_distillation.teacher_generation.packet_builder import build_packet_pair

# Every band of every rubric must hold at least this share of the dataset.
# Chosen below the worst observed share in the fitted design (~8.1% for
# engagement's score=1) so the test has headroom, not so it can't fail.
MIN_BAND_SHARE = 0.05

_BLUEPRINT_ORDER = {
    "intonation": ("flat", "moderate", "expressive"),
    "engagement": ("disengaged", "neutral", "engaging"),
}


@unittest.skipUnless(DEFAULT_PACKETS_PATH.exists(), f"{DEFAULT_PACKETS_PATH} not present in this checkout")
class DeliveryScoreDistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [json.loads(line) for line in Path(DEFAULT_PACKETS_PATH).read_text().splitlines() if line.strip()]
        cls.n = len(rows)
        cls.intonation_scores = Counter()
        cls.engagement_scores = Counter()
        cls.intonation_by_blueprint = {level: Counter() for level in _BLUEPRINT_ORDER["intonation"]}
        cls.engagement_by_blueprint = {level: Counter() for level in _BLUEPRINT_ORDER["engagement"]}

        for row in rows:
            pair = build_packet_pair(row)
            result = synthesize_delivery(pair.delivery, row["session_id"])
            intonation_score = result["scores"]["intonation"]["score"]
            engagement_score = result["scores"]["engagement"]["score"]

            cls.intonation_scores[intonation_score] += 1
            cls.engagement_scores[engagement_score] += 1
            cls.intonation_by_blueprint[row["blueprint"]["intonation"]][intonation_score] += 1
            cls.engagement_by_blueprint[row["blueprint"]["engagement"]][engagement_score] += 1

    def test_every_intonation_band_has_minimum_share(self):
        for band in (1, 2, 3, 4, 5):
            share = self.intonation_scores.get(band, 0) / self.n
            self.assertGreaterEqual(
                share, MIN_BAND_SHARE,
                f"intonation score={band} is only {share:.1%} of the dataset ({self.intonation_scores})",
            )

    def test_every_engagement_band_has_minimum_share(self):
        for band in (1, 2, 3, 4, 5):
            share = self.engagement_scores.get(band, 0) / self.n
            self.assertGreaterEqual(
                share, MIN_BAND_SHARE,
                f"engagement score={band} is only {share:.1%} of the dataset ({self.engagement_scores})",
            )

    def test_intonation_score_is_monotonic_with_blueprint(self):
        """flat < moderate < expressive on mean score -- this is the
        blueprint-inertness check: a rubric that doesn't correlate with
        the dimension it claims to measure is exactly the original bug
        (engagement's blueprint had zero effect on its own score)."""
        means = {
            level: sum(score * count for score, count in counter.items()) / sum(counter.values())
            for level, counter in self.intonation_by_blueprint.items()
            if sum(counter.values())
        }
        self.assertLess(means["flat"], means["moderate"])
        self.assertLess(means["moderate"], means["expressive"])

    def test_engagement_score_is_monotonic_with_blueprint(self):
        means = {
            level: sum(score * count for score, count in counter.items()) / sum(counter.values())
            for level, counter in self.engagement_by_blueprint.items()
            if sum(counter.values())
        }
        self.assertLess(means["disengaged"], means["neutral"])
        self.assertLess(means["neutral"], means["engaging"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
