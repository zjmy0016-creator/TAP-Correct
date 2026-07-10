from pathlib import Path
import unittest


class TestRequirements(unittest.TestCase):
    def test_analysis_script_dependencies_are_declared(self):
        requirements = {
            line.strip().lower()
            for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

        self.assertIn("scikit-learn", requirements)
        self.assertIn("scipy", requirements)


if __name__ == "__main__":
    unittest.main()
