import re
import unittest
from pathlib import Path


class ResponsiveBreakpointTests(unittest.TestCase):
    def test_ultra_wide_container_breakpoints_are_defined(self) -> None:
        responsive_css = Path("ragtime/frontend/src/styles/responsive.css").read_text()

        expected_breakpoints = {
            "1600": "1536px",
            "1920": "1720px",
        }
        for min_width, max_width in expected_breakpoints.items():
            pattern = re.compile(
                rf"@media\s*\(min-width:\s*{min_width}px\)\s*\{{\s*"
                rf":root\s*\{{\s*--container-max-width:\s*{re.escape(max_width)};\s*\}}\s*\}}",
                re.MULTILINE,
            )
            self.assertRegex(responsive_css, pattern)


if __name__ == "__main__":
    unittest.main()
