import hashlib
import sys
import unittest
from pathlib import Path

from src.generation.prompts import EXAMPLE_PRIORS, P1_GUIDELINES, PROMPTS, build_prompt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_prior_outputs import compare  # noqa: E402

EXAMPLES = ["A door opens.", "A car stops.", "A man sits."]


class VariantPromptTests(unittest.TestCase):
    def build(self, prior):
        return build_prompt(prior, 75.5, 4, 2.0, 12, "speech", EXAMPLES if prior in EXAMPLE_PRIORS else ())

    def test_frozen_p1_is_unchanged(self):
        expected = (P1_GUIDELINES + "Example 1: A door opens.\nExample 2: A car stops.\nExample 3: A man sits."
                    "\n\nCurrent video:\n" + self.build("P0"))
        self.assertEqual(self.build("P1"), expected)

    def test_each_variant_shares_the_current_video_text_and_differs_only_in_its_prefix(self):
        p0 = self.build("P0")
        prompts = {name: self.build(name) for name in PROMPTS}
        self.assertEqual(len(set(prompts.values())), len(prompts))
        self.assertTrue(all(text.endswith(p0) for text in prompts.values()))
        self.assertNotIn("Example 1", prompts["P1G"])
        self.assertNotIn("Professional descriptions", prompts["P1G"])
        self.assertIn("spatial relationships", prompts["P1G"])
        self.assertNotIn("spatial relationships", prompts["P1E"])
        self.assertIn("Example 3", prompts["P1E"])
        self.assertIn("one short sentence", prompts["P1S"])
        self.assertNotIn("spatial relationships", prompts["P1S"])

    def test_examples_are_required_or_forbidden_per_variant(self):
        for name in PROMPTS:
            with self.subTest(prior=name), self.assertRaises(ValueError):
                build_prompt(name, 1, 4, 2, 10, exemplars=() if name in EXAMPLE_PRIORS else EXAMPLES)


def row(moment, text, limit=10, **extra):
    return {"id": moment, "output": text, "word_limit": limit, **extra}


class CompareTests(unittest.TestCase):
    def test_pairs_by_id_and_keeps_over_budget_decodes(self):
        base = [row("a", "A man sits.", 4), row("b", "A dog runs.", 20), row("c", "x", hit_token_limit=True, error="cut")]
        long = "Beth sits down. She looks at the web."
        variant = [row("a", long, 4, error="word_limit_exceeded"), row("b", "A dog runs.", 20), row("c", "A cat.")]
        report = compare(base, variant)
        self.assertEqual((report["paired"], report["unpaired_incomplete"]), (2, 1))
        self.assertEqual(report["variant"]["over_budget"], 1)
        self.assertEqual(report["baseline"]["over_budget"], 0)
        self.assertEqual(report["variant"]["captions_with_leak_terms"], 1)
        self.assertEqual(report["baseline"]["captions_with_leak_terms"], 0)
        self.assertEqual(report["largest_length_increases"][0]["id"], "a")
        self.assertEqual(report["by_budget"][0]["variant_words"], 8)

    def test_refuses_to_compare_different_inputs(self):
        with self.assertRaises(ValueError):
            compare([row("a", "x", 4)], [row("a", "y", 9)])


if __name__ == "__main__":
    unittest.main()
