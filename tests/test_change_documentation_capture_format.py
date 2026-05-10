"""Tests for change documentation capture markdown normalization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_change_documentation_capture import _normalize_change_doc


class ChangeDocumentationCaptureFormatTests(unittest.TestCase):
    def test_normalize_places_bullets_directly_below_title_and_date(self) -> None:
        raw = """## What changed

- Updated `scripts/private_dashboard_server.py` with refreshed tabs.
- Added compare panel controls for scoring.

**Why it matters:**
- Improves quick review flow.
"""
        out = _normalize_change_doc(
            raw,
            changed_path="scripts/private_dashboard_server.py",
            trigger_event="afterFileEdit",
            timestamp_iso="2026-05-10T23:12:00Z",
        )
        lines = out.splitlines()
        self.assertEqual(lines[0], "## Change Documentation Update")
        self.assertEqual(lines[1], "Date: 2026-05-10")
        self.assertTrue(lines[2].startswith("- "))
        self.assertTrue(lines[3].startswith("- "))
        self.assertIn("Why it matters", out)

    def test_normalize_adds_fallback_summary_when_no_bullets_exist(self) -> None:
        raw = """## What changed

Updated docs and scripts for better benchmark reproducibility.
"""
        out = _normalize_change_doc(
            raw,
            changed_path="docs/WORKFLOW.md",
            trigger_event="afterFileEdit",
            timestamp_iso="2026-05-10T23:12:00Z",
        )
        lines = out.splitlines()
        self.assertEqual(lines[0], "## Change Documentation Update")
        self.assertEqual(lines[1], "Date: 2026-05-10")
        self.assertTrue(lines[2].startswith("- "))


if __name__ == "__main__":
    unittest.main()
