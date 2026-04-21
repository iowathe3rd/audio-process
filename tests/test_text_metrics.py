from __future__ import annotations
import unittest

from app.pipeline.stages.text.metrics import (
    deterministic_light_enhance,
    has_word_sequence_drift,
    language_switching_ratio,
)


class TextMetricsTests(unittest.TestCase):
    def test_language_switching_detects_kk_ru_mix(self) -> None:
        texts = ["сәлем как дела", "мен қазір приду"]
        ratio = language_switching_ratio(texts)
        self.assertGreater(ratio, 0.0)

    def test_deterministic_enhance_is_lexically_safe(self) -> None:
        source = "  алло   милиция  да   это  "
        enhanced = deterministic_light_enhance(source)
        self.assertEqual(enhanced, "Алло милиция да это")
        self.assertFalse(has_word_sequence_drift(source, enhanced))

    def test_word_sequence_drift_flags_lexical_change(self) -> None:
        source = "ну нашу курят это же противозаконно"
        candidate = "ну нашу крадут это же противозаконно"
        self.assertTrue(has_word_sequence_drift(source, candidate))


if __name__ == "__main__":
    unittest.main()
