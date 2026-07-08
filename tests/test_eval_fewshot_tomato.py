import unittest
from pathlib import Path


class TestFewshotTomatoConsoleOutput(unittest.TestCase):
    def test_runtime_status_literals_are_ascii(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "eval_fewshot_tomato.py"
        text = script.read_text(encoding="utf-8")
        forbidden = ["±", "🔍", "✅", "➖", "⚠️"]

        for marker in forbidden:
            self.assertNotIn(marker, text)


if __name__ == "__main__":
    unittest.main()
