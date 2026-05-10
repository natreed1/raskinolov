## Change Documentation Update
Date: 2026-05-10
- Added `tests/test_change_documentation_capture_format.py` to validate `docs/CHANGES.md` updates.
- Updated `docs/WORKFLOW.md` to call out `scripts/build_dataset.py` for new dataset captures.
- Added `scripts/build_dataset.py` to build and capture dataset YAML for `docs/DATA_CAPTURE.md`.
**Title:** Document dataset capture for `docs/CHANGES.md` on file edits.  
**Date:** 2023-11-11  
**Summary:**  
- Added `tests/test_change_documentation_capture_format.py` to validate `docs/CHANGES.md` updates.  
- Updated `docs/WORKFLOW.md` to call out `scripts/build_dataset.py` for new dataset captures.  
- Added `scripts/build_dataset.py` to build and capture dataset YAML for `docs/DATA_CAPTURE.md`.  
- Updated `docs/README.md` to call out `scripts/build_dataset.py` for new dataset captures.  

**What changed:**  
- Added `tests/test_change_documentation_capture_format.py` to validate `docs/CHANGES.md` updates.  
- Updated `docs/WORKFLOW.md` to call out `scripts/build_dataset.py` for new dataset captures.  
- Added `scripts/build_dataset.py` to build and capture dataset YAML for `docs/DATA_CAPTURE.md`.  
- Updated `docs/README.md` to call out `scripts/build_dataset.py` for new dataset captures.  

**Why it matters:**  
- Ensures `docs/CHANGES.md` is updated on file edits.  
- Provides a single command to capture new dataset YAML.  
- Reduces manual steps and risk of human error.  

**Commands/tests run:**  
- `python scripts/build_dataset.py` builds and captures dataset YAML for `docs/DATA_CAPTURE.md`.  
- `pytest tests/test_change_documentation_capture_format.py` validates `docs/CHANGES.md` updates.  

**Next validation steps:**  
- Run `python scripts/build_dataset.py` after each dataset capture.  
- Validate `docs/DATA_CAPTURE.md` contains expected YAML.  
- Run `pytest tests/test_change_documentation_capture_format.py` after each `docs/CHANGES.md` update.
