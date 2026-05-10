## 2026-05-05 — Dataset capture automation + dashboard refresh

**What changed:** 
- **Dataset capture:** `scripts/trigger_doc_training_on_changes.py` now spawns async documentation captures on watched edits, writes artifacts to `data/documentation_captures/`, and triggers dataset rebuilds. Heavy refreshes remain cooldown-gated.
- **Dashboard:** `scripts/private_dashboard_server.py` UI refreshed with tabs, blue/white theme, and dark mode. Added compare panel, feedback mode, and token savings estimates.

**Why it matters:** 
- **Automation:** Captures run on every edit, enabling iterative improvement. Heavy refreshes are still gated to preserve performance.
- **UX:** Modernized dashboard with tabs, compare panel, and feedback mode for clearer insights and actionable outputs.

**Commands/tests run:** 
- `python scripts/trigger_doc_training_on_changes.py` (simulated edit to trigger capture)
- `python scripts/private_dashboard_server.py` (run dashboard locally)
- `.venv/bin/python -m py_compile scripts/private_dashboard_server.py` (verify script compiles)

**Next validation steps:** 
- Verify captured artifacts appear in `data/documentation_captures/` and trigger dataset rebuilds.
- Confirm dashboard renders expected tabs, compare panel, and feedback mode.
