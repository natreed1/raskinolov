## Change Documentation Update
Date: 2026-05-10
- Fixed bug in CSV header detection.
- Added support for JSONL format.
- Updated README with new flags.
**Title:** Dataset Capture Script Update

**Date:** 2023-11-15

**Summary:** 
- Fixed bug in CSV header detection.
- Added support for JSONL format.
- Updated README with new flags.

**What changed:** 
- `scripts/generate_change_documentation_capture.py` now handles JSONL files and detects CSV headers more robustly.
- README was updated to include new flags and usage examples.

**Why it matters:** 
- Enhanced script functionality for diverse dataset formats.
- Improved user guidance through updated documentation.

**Commands/tests run:** 
- `python scripts/generate_change_documentation_capture.py` runs the script.
- `pytest tests/capture/` runs relevant tests.

**Next validation steps:** 
- Verify script works with JSONL files.
- Confirm README updates are clear and accurate.
