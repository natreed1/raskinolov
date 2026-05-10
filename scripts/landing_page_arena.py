#!/usr/bin/env python3
"""
Human evaluation arena for standalone Fallen Empire landing-page trials.

This intentionally does not mutate the game repo. It creates isolated static-site
attempt folders under benchmarks/results/landing_page_trials/ so local LoRA,
base, and Cursor/frontier outputs can be viewed, played with, and rated.

Examples:
  python scripts/landing_page_arena.py brief
  python scripts/landing_page_arena.py create --task "Design a Fallen Empire landing page"
  python scripts/landing_page_arena.py cursor-packet --trial-id <id> --attempt frontier_cursor
  python scripts/landing_page_arena.py scaffold --trial-id <id> --attempt local_manual
  python scripts/landing_page_arena.py local-attempt --trial-id <id> --attempt local_7b --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428
  python scripts/landing_page_arena.py serve --trial-id <id> --attempt local_300 --port 8091
  python scripts/landing_page_arena.py rate --trial-id <id> --attempt local_300 --visual-quality 4 --winner yes
  python scripts/landing_page_arena.py ui
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fe_lineage as _fe
from model_router import ChatMessage, GenerationRequest, LocalMlxBackend, OpenAICompatibleBackend

TRIALS_ROOT = REPO / "benchmarks" / "results" / "landing_page_trials"
TRIALS_INDEX = REPO / "benchmarks" / "results" / "landing_page_trials.jsonl"
DEFAULT_LOCAL_ADAPTER = os.environ.get("ADAPTER_PATH", _fe.DEFAULT_ARENA_ADAPTER_RELPATH)
COMPARISONS_INDEX = REPO / "benchmarks" / "results" / "landing_page_comparisons.jsonl"
TRAINING_DATA_INDEX = REPO / "benchmarks" / "results" / "landing_page_training_data.jsonl"
DEFAULT_BRIEF = REPO / "benchmarks" / "landing_page_brief.md"
DEFAULT_TASK = (
    "Create a polished standalone landing page for Fallen Empire, a hard-sci-fi "
    "strategy simulation game. The page should communicate the fantasy, show core "
    "systems, include faction/world flavor, and feel like a shippable website."
)
DEFAULT_ARENA_MAX_TOKENS = 8192

SAFE_STATIC_FILENAMES = {"index.html", "styles.css", "script.js"}
FENCE_RE = re.compile(
    r"```(?P<info>[^\n`]*)\n(?P<body>.*?)```",
    re.DOTALL,
)
HTML_LINK_RE = re.compile(
    r"(?P<prefix>\b(?:href|src)\s*=\s*)(?P<quote>[\"'])(?P<target>(?!https?:|mailto:|tel:|#|/)[^\"'#?]+\.html)(?P<rest>[^\"']*)(?P=quote)",
    re.IGNORECASE,
)
ARENA_CSS = """
.gradio-container {
  background: radial-gradient(circle at 15% 0%, #152238 0, transparent 32%),
    linear-gradient(135deg, #05070c 0%, #0b1019 54%, #07090f 100%);
  color: #dce7f7;
}
.fe-shell {
  border: 1px solid rgba(125, 166, 255, 0.18);
  border-radius: 18px;
  padding: 16px;
  background: rgba(9, 13, 22, 0.72);
  box-shadow: 0 22px 70px rgba(0, 0, 0, 0.35);
}
/* Gradio Group's inner flex uses overflow:hidden, which can swallow pointer events for controls below tall previews. */
.gr-group.fe-shell > div {
  overflow: visible !important;
}
.fe-preview {
  width: 100%;
  height: 620px;
  border: 1px solid rgba(143, 179, 255, 0.26);
  border-radius: 14px;
  background: #05070c;
}
textarea, .cm-editor {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace !important;
}
"""


def load_dotenv(path: Path = REPO / ".env") -> None:
    """Load simple KEY=VALUE lines without overwriting exported env vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Trial:
    trial_id: str
    task: str
    created_at: str
    brief_path: str
    trial_dir: str


@dataclass
class Rating:
    trial_id: str
    attempt: str
    rated_at: str
    visual_quality: int
    game_fit: int
    copy_quality: int
    responsiveness: int
    interaction_quality: int
    mergeability: int
    cleanup_minutes: int
    winner: bool
    notes: str


@dataclass
class ComparisonRecord:
    trial_id: str
    saved_at: str
    task: str
    local_attempt: str
    frontier_attempt: str
    winner: str
    local_scores: Dict[str, int]
    frontier_scores: Dict[str, int]
    local_notes: str
    frontier_notes: str
    training_value: str
    local_path: str
    frontier_path: str


def _utc_id(prefix: str = "trial") -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_{prefix}"


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:60] or "attempt"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_trial(trial_id: str) -> Trial:
    trial_dir = TRIALS_ROOT / trial_id
    meta_path = trial_dir / "trial.json"
    if not meta_path.is_file():
        raise SystemExit(f"Trial not found: {trial_id} ({meta_path})")
    return Trial(**json.loads(meta_path.read_text(encoding="utf-8")))


def _attempt_dir(trial_id: str, attempt: str) -> Path:
    return TRIALS_ROOT / trial_id / "attempts" / _slug(attempt)


def _list_attempt_dirs(trial_id: str) -> List[Path]:
    attempts_root = TRIALS_ROOT / trial_id / "attempts"
    if not attempts_root.is_dir():
        return []
    return sorted(p for p in attempts_root.iterdir() if p.is_dir())


def build_default_brief() -> str:
    project_state = _read(REPO / "docs" / "PROJECT_STATE.md")
    workflow = _read(REPO / "docs" / "WORKFLOW.md")
    readme = _read(REPO / "README.md")
    return f"""# Fallen Empire Landing Page Brief

## Product Intent

Fallen Empire is a hard-sci-fi strategy simulation game about imperial collapse,
frontier logistics, faction pressure, and systemic consequences. The landing page
should sell the fantasy of commanding a brittle empire where every tactical choice
feeds back into morale, economy, territory, and survival.

## Desired Visual Direction

- Dark, high-contrast sci-fi interface.
- Tactical map / command center mood.
- Clear readable typography; avoid generic fantasy ornament.
- Use sections that could plausibly become a real game website:
  hero, feature cards, simulation systems, factions/world, screenshots/placeholders,
  and a call-to-action.
- Responsive layout for desktop and mobile.

## Evaluation Criteria

Rate outputs by actually viewing the page, not just reading code:

- Visual quality and hierarchy.
- How well the copy understands Fallen Empire.
- Whether the page feels specific rather than generic.
- Responsiveness and interaction polish.
- Ease of using or extending the code.

## Repository Context Excerpts

### README

```markdown
{readme[:3000]}
```

### Project State

```markdown
{project_state[:6000]}
```

### ML Workflow

```markdown
{workflow[:4000]}
```
"""


def write_brief(path: Path = DEFAULT_BRIEF, overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        return path
    _write(path, build_default_brief())
    return path


def create_trial(task: str, brief_path: Path = DEFAULT_BRIEF, trial_id: Optional[str] = None) -> Trial:
    brief = write_brief(brief_path)
    tid = trial_id or _utc_id(_slug(task)[:24])
    trial_dir = TRIALS_ROOT / tid
    trial_dir.mkdir(parents=True, exist_ok=False)
    _write(trial_dir / "task.md", f"# Trial Task\n\n{task.strip()}\n")
    shutil.copyfile(brief, trial_dir / "brief.md")
    trial = Trial(
        trial_id=tid,
        task=task.strip(),
        created_at=datetime.now(timezone.utc).isoformat(),
        brief_path=str(brief.relative_to(REPO)),
        trial_dir=str(trial_dir.relative_to(REPO)),
    )
    _write(trial_dir / "trial.json", json.dumps(asdict(trial), indent=2) + "\n")
    _append_jsonl(TRIALS_INDEX, asdict(trial))
    return trial


def cursor_packet(trial_id: str, attempt: str = "frontier_cursor") -> Path:
    trial = _load_trial(trial_id)
    attempt_path = _attempt_dir(trial_id, attempt)
    task = _read(Path(trial.trial_dir) if Path(trial.trial_dir).is_absolute() else REPO / trial.trial_dir / "task.md")
    brief = _read(REPO / trial.trial_dir / "brief.md")
    text = f"""# Cursor Frontier Task Packet

You are creating an isolated static landing-page attempt for a model comparison trial.
Do not edit the original game repo. Produce files inside this attempt directory:

`{attempt_path}`

## Task

{task}

## Required Output

Create a standalone static website with:

- `index.html`
- `styles.css` if useful
- `script.js` if useful
- no external paid services
- no build step required

The page should be visually evaluable in a browser via:

```bash
python -m http.server 8090 --directory "{attempt_path}"
```

## Human Evaluation Criteria

- Visual quality
- Game/product fit
- Copywriting
- Responsive layout
- Interaction polish
- How much cleanup is needed

## Context

{brief}
"""
    _write(attempt_path / "cursor_packet.md", text)
    _write(attempt_path / "README.md", _attempt_readme(trial_id, attempt))
    return attempt_path / "cursor_packet.md"


def _attempt_readme(trial_id: str, attempt: str) -> str:
    return f"""# Landing Page Attempt `{attempt}`

Trial: `{trial_id}`

Preview:

```bash
python -m http.server 8090 --directory "{_attempt_dir(trial_id, attempt)}"
```
"""


def scaffold_attempt(trial_id: str, attempt: str) -> Path:
    attempt_path = _attempt_dir(trial_id, attempt)
    attempt_path.mkdir(parents=True, exist_ok=True)
    _write(attempt_path / "README.md", _attempt_readme(trial_id, attempt))
    _write(
        attempt_path / "index.html",
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fallen Empire Landing Page Attempt</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <main class="shell">
    <p class="eyebrow">Fallen Empire</p>
    <h1>Command the collapse before it commands you.</h1>
    <p class="lead">This scaffold is ready for a model or human to replace with a richer landing page.</p>
  </main>
</body>
</html>
""",
    )
    _write(
        attempt_path / "styles.css",
        """body {
  margin: 0;
  min-height: 100vh;
  display: grid;
  place-items: center;
  background: #080b12;
  color: #e5edf8;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.shell { max-width: 780px; padding: 48px; }
.eyebrow { color: #8fb3ff; text-transform: uppercase; letter-spacing: 0.18em; }
h1 { font-size: clamp(42px, 8vw, 88px); line-height: 0.92; margin: 0 0 24px; }
.lead { color: #aab6c7; font-size: 20px; line-height: 1.6; }
""",
    )
    return attempt_path


def _generation_prompt(trial: Trial) -> str:
    brief = _read(REPO / trial.trial_dir / "brief.md")
    task = _read(REPO / trial.trial_dir / "task.md")
    return f"""You are producing a standalone static landing page for a model comparison trial.

Return the complete files as fenced code blocks. Use this exact format for every file:

```html path=index.html
...
```

```css path=styles.css
...
```

```javascript path=script.js
...
```

Output rules:

- Return only these file blocks: `index.html`, `styles.css`, and `script.js`.
- Use `index.html` as the only HTML page. Do not link to `game.html`, `demo.html`, or any other local HTML file.
- Do not include Markdown commentary outside the fenced file blocks.
- No build step. No external paid services. The result must be viewable with `python -m http.server`.

{task}

{brief}
"""


def _candidate_path_from_fence(info: str) -> Optional[str]:
    """Recover a filename from common model fence styles."""
    info = info.strip()
    if not info:
        return None
    language_defaults = {
        "html": "index.html",
        "css": "styles.css",
        "js": "script.js",
        "javascript": "script.js",
    }
    if info.lower() in language_defaults:
        return language_defaults[info.lower()]
    for key in ("path", "file", "filename", "name"):
        m = re.search(rf"{key}\s*=\s*(?P<quote>[\"']?)(?P<path>[^\"'\s,;]+)(?P=quote)", info, re.IGNORECASE)
        if m:
            return m.group("path")
    for token in re.split(r"[\s,;]+", info):
        clean = token.strip().strip('"').strip("'").strip("`")
        if clean.endswith((".html", ".css", ".js")):
            return clean
    return None


def _safe_attempt_path(raw_path: str) -> Optional[str]:
    rel = raw_path.strip().strip('"').strip("'").strip("`")
    rel = rel.removeprefix("./")
    if rel in {"game.html", "landing.html", "demo.html", "home.html", "site.html"}:
        rel = "index.html"
    if rel.startswith("/") or ".." in Path(rel).parts:
        return None
    if "/" in rel or "\\" in rel:
        return None
    if rel not in SAFE_STATIC_FILENAMES:
        return None
    return rel


def _extract_labeled_sections(markdown: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    section = re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:file|filename|path)?\s*[:\-]?\s*`?(?P<path>index\.html|styles\.css|script\.js|game\.html|landing\.html|demo\.html)`?\s*$"
    )
    matches = list(section.finditer(markdown))
    for idx, match in enumerate(matches):
        rel = _safe_attempt_path(match.group("path"))
        if not rel:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        if body.startswith("```"):
            fenced = FENCE_RE.search(body)
            if fenced:
                body = fenced.group("body")
        if body:
            files[rel] = body.strip() + "\n"
    return files


def _normalize_generated_files(files: Dict[str, str]) -> Dict[str, str]:
    normalized = {rel: body for rel, body in files.items() if rel in SAFE_STATIC_FILENAMES}
    html_body = normalized.get("index.html", "")
    if html_body:
        generated_html = {rel for rel in normalized if rel.endswith(".html")}

        def replace_missing(match: re.Match[str]) -> str:
            target = match.group("target")
            if target in generated_html:
                return match.group(0)
            quote = match.group("quote")
            return f"{match.group('prefix')}{quote}#{quote}"

        normalized["index.html"] = HTML_LINK_RE.sub(replace_missing, html_body)
    return normalized


def _extract_files(markdown: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for m in FENCE_RE.finditer(markdown):
        rel = _candidate_path_from_fence(m.group("info"))
        if not rel:
            continue
        safe_rel = _safe_attempt_path(rel)
        if not safe_rel:
            continue
        files[safe_rel] = m.group("body").strip() + "\n"
    if not files:
        files = _extract_labeled_sections(markdown)
    if not files and "<html" in markdown.lower():
        files["index.html"] = markdown.strip() + "\n"
    return _normalize_generated_files(files)


def write_model_attempt(
    trial_id: str,
    attempt: str,
    model_output: str,
    route: str,
    adapter_path: Optional[str],
) -> Path:
    attempt_path = _attempt_dir(trial_id, attempt)
    attempt_path.mkdir(parents=True, exist_ok=True)
    _write(attempt_path / "README.md", _attempt_readme(trial_id, attempt))
    _write(attempt_path / "model_output.md", model_output)
    files = _extract_files(model_output)
    if not files:
        _write(
            attempt_path / "parse_error.html",
            "<!doctype html><meta charset='utf-8'><title>Unparsed Model Output</title>"
            "<style>body{font-family:system-ui;margin:40px;white-space:pre-wrap;background:#090b10;color:#eef}</style>"
            f"<h1>Model output was not parseable as files</h1><pre>{html.escape(model_output)}</pre>",
        )
    else:
        for rel, body in files.items():
            _write(attempt_path / rel, body)
    meta = {
        "trial_id": trial_id,
        "attempt": attempt,
        "route": route,
        "adapter_path": adapter_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": sorted(files.keys()),
        "parse_ok": bool(files),
    }
    _write(attempt_path / "attempt.json", json.dumps(meta, indent=2) + "\n")
    return attempt_path


def local_attempt(
    trial_id: str,
    attempt: str,
    adapter_path: Optional[str],
    model_id: str,
    max_tokens: int,
    temp: float,
) -> Path:
    trial = _load_trial(trial_id)
    backend = LocalMlxBackend(model_id=model_id, adapter_path=adapter_path)
    request = GenerationRequest(
        messages=[
            ChatMessage("system", "You are a senior product-minded frontend engineer. Return complete static files."),
            ChatMessage("user", _generation_prompt(trial)),
        ],
        max_tokens=max_tokens,
        temperature=temp,
        force_route="local",
    )
    text = backend.generate(request)
    return write_model_attempt(trial_id, attempt, text, "local_mlx", adapter_path)


def frontier_attempt(
    trial_id: str,
    attempt: str,
    model: Optional[str],
    max_tokens: int,
    temp: float,
    base_url: Optional[str] = None,
) -> Path:
    trial = _load_trial(trial_id)
    backend = OpenAICompatibleBackend(base_url=base_url or None, model=model or None)
    request = GenerationRequest(
        messages=[
            ChatMessage("system", "You are a senior product-minded frontend engineer. Return complete static files."),
            ChatMessage("user", _generation_prompt(trial)),
        ],
        max_tokens=max_tokens,
        temperature=temp,
        force_route="frontier",
    )
    text, usage = backend.generate(request)
    attempt_path = write_model_attempt(trial_id, attempt, text, "openai_compatible", None)
    _write(attempt_path / "frontier_usage.json", json.dumps(usage, indent=2) + "\n")
    return attempt_path


def import_attempt(
    trial_id: str,
    attempt: str,
    model_output: str,
    source: str,
) -> Path:
    return write_model_attempt(
        trial_id=trial_id,
        attempt=attempt,
        model_output=model_output,
        route=source or "imported",
        adapter_path=None,
    )


def _attempt_file_text(trial_id: str, attempt: str, rel: str) -> str:
    return _read(_attempt_dir(trial_id, attempt) / rel)


def _attempt_metadata(trial_id: str, attempt: str) -> dict:
    meta = _read(_attempt_dir(trial_id, attempt) / "attempt.json")
    if not meta:
        return {}
    try:
        return json.loads(meta)
    except json.JSONDecodeError:
        return {}


def _attempt_training_payload(trial_id: str, attempt: str) -> Dict[str, str]:
    return {
        "index.html": _attempt_file_text(trial_id, attempt, "index.html"),
        "styles.css": _attempt_file_text(trial_id, attempt, "styles.css"),
        "script.js": _attempt_file_text(trial_id, attempt, "script.js"),
        "model_output.md": _attempt_file_text(trial_id, attempt, "model_output.md"),
    }


def read_attempt_files(trial_id: str, attempt: str) -> Tuple[str, str, str]:
    return (
        _attempt_file_text(trial_id, attempt, "index.html"),
        _attempt_file_text(trial_id, attempt, "styles.css"),
        _attempt_file_text(trial_id, attempt, "script.js"),
    )


def save_attempt_files(
    trial_id: str,
    attempt: str,
    html_text: str,
    css_text: str,
    js_text: str,
    source: str = "human_edit",
) -> Path:
    attempt_path = scaffold_attempt(trial_id, attempt)
    _write(attempt_path / "index.html", html_text or "")
    _write(attempt_path / "styles.css", css_text or "")
    _write(attempt_path / "script.js", js_text or "")
    meta = {
        "trial_id": trial_id,
        "attempt": attempt,
        "source": source,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "files": ["index.html", "styles.css", "script.js"],
    }
    _write(attempt_path / "attempt.json", json.dumps(meta, indent=2) + "\n")
    return attempt_path


def save_comparison_record(
    trial_id: str,
    local_attempt_name: str,
    frontier_attempt_name: str,
    winner: str,
    local_scores: Dict[str, int],
    frontier_scores: Dict[str, int],
    local_notes: str,
    frontier_notes: str,
    training_value: str,
) -> Path:
    trial = _load_trial(trial_id)
    local_path = _attempt_dir(trial_id, local_attempt_name)
    frontier_path = _attempt_dir(trial_id, frontier_attempt_name)
    record = ComparisonRecord(
        trial_id=trial_id,
        saved_at=datetime.now(timezone.utc).isoformat(),
        task=trial.task,
        local_attempt=local_attempt_name,
        frontier_attempt=frontier_attempt_name,
        winner=winner,
        local_scores=local_scores,
        frontier_scores=frontier_scores,
        local_notes=local_notes,
        frontier_notes=frontier_notes,
        training_value=training_value,
        local_path=str(local_path.relative_to(REPO)),
        frontier_path=str(frontier_path.relative_to(REPO)),
    )
    trial_dir = TRIALS_ROOT / trial_id
    row = asdict(record)
    row["brief"] = _read(trial_dir / "brief.md")
    row["local_files"] = _attempt_training_payload(trial_id, local_attempt_name)
    row["frontier_files"] = _attempt_training_payload(trial_id, frontier_attempt_name)
    out = trial_dir / "comparison.json"
    _write(out, json.dumps(row, indent=2, ensure_ascii=False) + "\n")
    _append_jsonl(COMPARISONS_INDEX, row)
    _append_jsonl(TRAINING_DATA_INDEX, row)
    return out


def rate_attempt(args: argparse.Namespace) -> Path:
    rating = Rating(
        trial_id=args.trial_id,
        attempt=args.attempt,
        rated_at=datetime.now(timezone.utc).isoformat(),
        visual_quality=args.visual_quality,
        game_fit=args.game_fit,
        copy_quality=args.copy_quality,
        responsiveness=args.responsiveness,
        interaction_quality=args.interaction_quality,
        mergeability=args.mergeability,
        cleanup_minutes=args.cleanup_minutes,
        winner=args.winner,
        notes=args.notes or "",
    )
    attempt_path = _attempt_dir(args.trial_id, args.attempt)
    out = attempt_path / "rating.json"
    _write(out, json.dumps(asdict(rating), indent=2) + "\n")
    _append_jsonl(TRIALS_INDEX.parent / "landing_page_ratings.jsonl", asdict(rating))
    return out


def serve_attempt(trial_id: str, attempt: str, port: int) -> None:
    attempt_path = _attempt_dir(trial_id, attempt)
    if not (attempt_path / "index.html").is_file():
        raise SystemExit(f"No index.html found for attempt: {attempt_path}")
    print(f"Serving {attempt_path}")
    print(f"Open http://127.0.0.1:{port}")
    subprocess.run([sys.executable, "-m", "http.server", str(port), "--directory", str(attempt_path)], check=False)


def list_trials() -> None:
    if not TRIALS_ROOT.is_dir():
        print("No landing-page trials yet.")
        return
    for trial_dir in sorted(TRIALS_ROOT.iterdir(), reverse=True):
        meta = trial_dir / "trial.json"
        if not meta.is_file():
            continue
        payload = json.loads(meta.read_text(encoding="utf-8"))
        attempts = sorted(p.name for p in (trial_dir / "attempts").glob("*") if p.is_dir()) if (trial_dir / "attempts").is_dir() else []
        print(f"{payload['trial_id']}  attempts={','.join(attempts) or '-'}")
        print(f"  {payload['task']}")


def build_app():
    import gradio as gr

    def create_ui(task: str):
        trial = create_trial(task or DEFAULT_TASK)
        packet = cursor_packet(trial.trial_id)
        return trial.trial_id, f"Created `{trial.trial_id}`\n\nCursor packet: `{packet}`"

    def _render_document(html_text: str, css_text: str, js_text: str) -> str:
        if not html_text.strip():
            return "<!doctype html><meta charset='utf-8'><body style='font-family:system-ui;background:#070a12;color:#dce7f7;padding:28px'>No HTML generated yet.</body>"
        injected = html_text
        if css_text.strip() and "</head>" in injected:
            injected = injected.replace("</head>", f"<style>\n{css_text}\n</style>\n</head>", 1)
        elif css_text.strip():
            injected = f"<style>\n{css_text}\n</style>\n{injected}"
        if js_text.strip() and "</body>" in injected:
            injected = injected.replace("</body>", f"<script>\n{js_text}\n</script>\n</body>", 1)
        elif js_text.strip():
            injected = f"{injected}\n<script>\n{js_text}\n</script>"
        return injected

    def _render_preview_frame(html_text: str, css_text: str, js_text: str) -> str:
        document = _render_document(html_text, css_text, js_text)
        return (
            "<iframe class='fe-preview' sandbox='allow-scripts allow-same-origin' "
            f"srcdoc=\"{html.escape(document, quote=True)}\"></iframe>"
        )

    def _attempt_result_line(trial_id: str, attempt: str, label: str) -> str:
        meta = _attempt_metadata(trial_id, attempt)
        if meta.get("parse_ok") is False:
            return (
                f"{label} response saved, but no file blocks were parsed. "
                f"Raw output is in `{_attempt_dir(trial_id, attempt) / 'model_output.md'}`."
            )
        files = meta.get("files") or []
        if files:
            return f"{label} generated `{attempt}` with {', '.join(files)}."
        return f"{label} prepared `{attempt}`."

    def ide_run_split_ui(
        task: str,
        local_adapter: str,
        frontier_model: str,
        frontier_base_url: str,
        run_frontier_api: bool,
        max_tokens: int,
        temp: float,
    ):
        trial = create_trial(task or DEFAULT_TASK)
        local_name = "local_300"
        frontier_name = "frontier_api" if run_frontier_api else "frontier_cursor"
        statuses: Dict[str, str] = {}

        def run_local_lane():
            path = local_attempt(
                trial_id=trial.trial_id,
                attempt=local_name,
                adapter_path=(local_adapter or "").strip() or None,
                model_id=os.environ.get("MODEL", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"),
                max_tokens=int(max_tokens),
                temp=float(temp),
            )
            return "local", _attempt_result_line(trial.trial_id, local_name, "Local")

        def run_frontier_lane():
            if run_frontier_api:
                path = frontier_attempt(
                    trial_id=trial.trial_id,
                    attempt=frontier_name,
                    model=(frontier_model or "").strip() or None,
                    base_url=(frontier_base_url or "").strip() or None,
                    max_tokens=int(max_tokens),
                    temp=float(temp),
                )
                return "frontier", _attempt_result_line(trial.trial_id, frontier_name, "Frontier")
            packet = cursor_packet(trial.trial_id, frontier_name)
            return "frontier", f"Cursor packet ready at `{packet}`. Paste result in the Frontier tab or code pane."

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_local_lane), pool.submit(run_frontier_lane)]
            for fut in as_completed(futures):
                try:
                    key, value = fut.result()
                except Exception as exc:
                    key, value = "error", f"{type(exc).__name__}: {exc}"
                statuses[key] = value

        l_html, l_css, l_js = read_attempt_files(trial.trial_id, local_name)
        f_html, f_css, f_js = read_attempt_files(trial.trial_id, frontier_name)
        status = (
            f"Trial `{trial.trial_id}`\n\n"
            f"- {statuses.get('local', 'Local lane did not run')}\n"
            f"- {statuses.get('frontier', statuses.get('error', 'Frontier lane did not run'))}"
        )
        return (
            trial.trial_id,
            local_name,
            frontier_name,
            status,
            l_html,
            l_css,
            l_js,
            f_html,
            f_css,
            f_js,
            _render_preview_frame(l_html, l_css, l_js),
            _render_preview_frame(f_html, f_css, f_js),
        )

    def ide_preview_ui(
        trial_id: str,
        local_attempt_name: str,
        frontier_attempt_name: str,
        local_html: str,
        local_css: str,
        local_js: str,
        frontier_html: str,
        frontier_css: str,
        frontier_js: str,
    ):
        trial_id = trial_id.strip()
        local_attempt_name = local_attempt_name.strip() or "local_300"
        frontier_attempt_name = frontier_attempt_name.strip() or "frontier_api"
        save_attempt_files(trial_id, local_attempt_name, local_html, local_css, local_js, source="ide_local_edit")
        save_attempt_files(trial_id, frontier_attempt_name, frontier_html, frontier_css, frontier_js, source="ide_frontier_edit")
        return (
            _render_preview_frame(local_html, local_css, local_js),
            _render_preview_frame(frontier_html, frontier_css, frontier_js),
            "Preview refreshed and current code saved to both attempt folders.",
        )

    def ide_save_training_ui(
        trial_id: str,
        local_attempt_name: str,
        frontier_attempt_name: str,
        local_html: str,
        local_css: str,
        local_js: str,
        frontier_html: str,
        frontier_css: str,
        frontier_js: str,
        winner: str,
        local_visual: int,
        local_fit: int,
        local_copy: int,
        local_responsive: int,
        local_interaction: int,
        local_merge: int,
        frontier_visual: int,
        frontier_fit: int,
        frontier_copy: int,
        frontier_responsive: int,
        frontier_interaction: int,
        frontier_merge: int,
        local_notes: str,
        frontier_notes: str,
        training_value: str,
    ):
        trial_id = trial_id.strip()
        local_attempt_name = local_attempt_name.strip() or "local_300"
        frontier_attempt_name = frontier_attempt_name.strip() or "frontier_api"
        save_attempt_files(trial_id, local_attempt_name, local_html, local_css, local_js, source="ide_local_final")
        save_attempt_files(trial_id, frontier_attempt_name, frontier_html, frontier_css, frontier_js, source="ide_frontier_final")
        out = save_comparison_record(
            trial_id=trial_id,
            local_attempt_name=local_attempt_name,
            frontier_attempt_name=frontier_attempt_name,
            winner=winner,
            local_scores={
                "visual_quality": int(local_visual),
                "game_fit": int(local_fit),
                "copy_quality": int(local_copy),
                "responsiveness": int(local_responsive),
                "interaction_quality": int(local_interaction),
                "mergeability": int(local_merge),
            },
            frontier_scores={
                "visual_quality": int(frontier_visual),
                "game_fit": int(frontier_fit),
                "copy_quality": int(frontier_copy),
                "responsiveness": int(frontier_responsive),
                "interaction_quality": int(frontier_interaction),
                "mergeability": int(frontier_merge),
            },
            local_notes=local_notes or "",
            frontier_notes=frontier_notes or "",
            training_value=training_value or "",
        )
        return f"Saved comparison and training data: `{out}`"

    def _preview_for(trial_id: str, attempt: str) -> str:
        index = _attempt_dir(trial_id, attempt) / "index.html"
        return _read(index) or f"<p>No previewable <code>index.html</code> for <code>{attempt}</code>.</p>"

    def compare_generate_ui(
        task: str,
        local_adapter: str,
        frontier_model: str,
        frontier_base_url: str,
        use_frontier_api: bool,
        max_tokens: int,
        temp: float,
    ):
        trial = create_trial(task or DEFAULT_TASK)
        local_name = "local_300"
        frontier_name = "frontier_api" if use_frontier_api else "frontier_cursor"
        statuses: Dict[str, str] = {}

        def run_local():
            path = local_attempt(
                trial_id=trial.trial_id,
                attempt=local_name,
                adapter_path=(local_adapter or "").strip() or None,
                model_id=os.environ.get("MODEL", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"),
                max_tokens=int(max_tokens),
                temp=float(temp),
            )
            return "local", f"Local complete: `{path}`"

        def run_frontier():
            if use_frontier_api:
                path = frontier_attempt(
                    trial_id=trial.trial_id,
                    attempt=frontier_name,
                    model=(frontier_model or "").strip() or None,
                    base_url=(frontier_base_url or "").strip() or None,
                    max_tokens=int(max_tokens),
                    temp=float(temp),
                )
                return "frontier", f"Frontier API complete: `{path}`"
            packet = cursor_packet(trial.trial_id, frontier_name)
            return "frontier", f"Frontier packet created: `{packet}`. Paste/import Cursor output in the Frontier tab to make it previewable."

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_local), pool.submit(run_frontier)]
            for fut in as_completed(futures):
                try:
                    key, value = fut.result()
                except Exception as exc:
                    key, value = "error", f"`{type(exc).__name__}: {exc}`"
                statuses[key] = value

        status = (
            f"Trial: `{trial.trial_id}`\n\n"
            f"- {statuses.get('local', 'Local did not run')}\n"
            f"- {statuses.get('frontier', statuses.get('error', 'Frontier did not run'))}"
        )
        return (
            trial.trial_id,
            local_name,
            frontier_name,
            status,
            _preview_for(trial.trial_id, local_name),
            _preview_for(trial.trial_id, frontier_name),
        )

    def compare_refresh_ui(trial_id: str, local_attempt_name: str, frontier_attempt_name: str):
        trial_id = trial_id.strip()
        return (
            _preview_for(trial_id, local_attempt_name.strip() or "local_300"),
            _preview_for(trial_id, frontier_attempt_name.strip() or "frontier_api"),
        )

    def compare_save_ui(
        trial_id: str,
        local_attempt_name: str,
        frontier_attempt_name: str,
        winner: str,
        local_visual: int,
        local_fit: int,
        local_copy: int,
        local_responsive: int,
        local_interaction: int,
        local_merge: int,
        frontier_visual: int,
        frontier_fit: int,
        frontier_copy: int,
        frontier_responsive: int,
        frontier_interaction: int,
        frontier_merge: int,
        local_notes: str,
        frontier_notes: str,
        training_value: str,
    ):
        out = save_comparison_record(
            trial_id=trial_id.strip(),
            local_attempt_name=local_attempt_name.strip() or "local_300",
            frontier_attempt_name=frontier_attempt_name.strip() or "frontier_api",
            winner=winner,
            local_scores={
                "visual_quality": int(local_visual),
                "game_fit": int(local_fit),
                "copy_quality": int(local_copy),
                "responsiveness": int(local_responsive),
                "interaction_quality": int(local_interaction),
                "mergeability": int(local_merge),
            },
            frontier_scores={
                "visual_quality": int(frontier_visual),
                "game_fit": int(frontier_fit),
                "copy_quality": int(frontier_copy),
                "responsiveness": int(frontier_responsive),
                "interaction_quality": int(frontier_interaction),
                "mergeability": int(frontier_merge),
            },
            local_notes=local_notes or "",
            frontier_notes=frontier_notes or "",
            training_value=training_value or "",
        )
        return f"Saved comparison and training data: `{out}`"

    def packet_ui(trial_id: str, attempt: str):
        packet = cursor_packet(trial_id.strip(), attempt or "frontier_cursor")
        return str(packet), _read(packet)

    def scaffold_ui(trial_id: str, attempt: str):
        path = scaffold_attempt(trial_id.strip(), attempt or "manual_attempt")
        return f"Scaffolded: `{path}`\nServe with: `python scripts/landing_page_arena.py serve --trial-id {trial_id.strip()} --attempt {attempt or 'manual_attempt'} --port 8090`"

    def local_ui(trial_id: str, attempt: str, adapter_path: str, max_tokens: int, temp: float):
        try:
            path = local_attempt(
                trial_id=trial_id.strip(),
                attempt=attempt.strip() or "local_300",
                adapter_path=(adapter_path or "").strip() or None,
                model_id=os.environ.get("MODEL", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"),
                max_tokens=int(max_tokens),
                temp=float(temp),
            )
            index = path / "index.html"
            return f"Generated local attempt: `{path}`", _read(index)
        except Exception as exc:
            return f"Local generation failed: `{type(exc).__name__}: {exc}`", ""

    def frontier_api_ui(trial_id: str, attempt: str, model: str, base_url: str, max_tokens: int, temp: float):
        try:
            path = frontier_attempt(
                trial_id=trial_id.strip(),
                attempt=attempt.strip() or "frontier_api",
                model=(model or "").strip() or None,
                base_url=(base_url or "").strip() or None,
                max_tokens=int(max_tokens),
                temp=float(temp),
            )
            index = path / "index.html"
            return f"Generated frontier attempt: `{path}`", _read(index)
        except Exception as exc:
            return (
                f"Frontier API generation failed: `{type(exc).__name__}: {exc}`\n\n"
                "Set `FRONTIER_API_KEY` or `OPENAI_API_KEY` in your shell before launching the UI. "
                "Set `FRONTIER_MODEL` to your Codex/frontier model id.",
                "",
            )

    def import_ui(trial_id: str, attempt: str, source: str, pasted: str):
        if not pasted.strip():
            return "Paste a model response or fenced file blocks first.", ""
        path = import_attempt(
            trial_id=trial_id.strip(),
            attempt=attempt.strip() or "frontier_paste",
            model_output=pasted,
            source=source.strip() or "frontier_paste",
        )
        return f"Imported attempt: `{path}`", _read(path / "index.html")

    def import_frontier_code_ui(trial_id: str, attempt: str, pasted: str):
        trial_id = trial_id.strip()
        attempt = attempt.strip() or "frontier_api"
        if not trial_id:
            return "Run the split first so the arena has a trial to save into.", "", "", "", ""
        if not pasted.strip():
            return "Paste a model response or fenced file blocks first.", "", "", "", ""
        path = import_attempt(
            trial_id=trial_id,
            attempt=attempt,
            model_output=pasted,
            source="frontier_paste",
        )
        html_text, css_text, js_text = read_attempt_files(trial_id, attempt)
        return (
            f"Imported frontier code into `{path}`",
            html_text,
            css_text,
            js_text,
            _render_preview_frame(html_text, css_text, js_text),
        )

    def attempts_ui(trial_id: str):
        trial_id = trial_id.strip()
        attempts = _list_attempt_dirs(trial_id)
        if not attempts:
            return (
                "No attempts yet. Generate a Cursor packet, scaffold an attempt, "
                "or run a local attempt first."
            )
        lines = [f"- `{p.name}`" + (" (previewable)" if (p / "index.html").is_file() else " (packet/output only)") for p in attempts]
        return "Available attempts:\n" + "\n".join(lines)

    def preview_ui(trial_id: str, attempt: str):
        trial_id = trial_id.strip()
        attempt = (attempt or "").strip()
        attempts = _list_attempt_dirs(trial_id)
        if not attempt:
            previewable = [p for p in attempts if (p / "index.html").is_file()]
            if not previewable:
                available = ", ".join(p.name for p in attempts) or "none"
                return (
                    "<p>No previewable <code>index.html</code> exists yet.</p>"
                    f"<p>Available attempts: <code>{available}</code></p>"
                    "<p>Use the Attempts tab to scaffold an attempt, run "
                    "<code>local-attempt</code>, or paste Cursor output into the attempt folder.</p>"
                )
            index = previewable[0] / "index.html"
        else:
            direct = _attempt_dir(trial_id, attempt)
            existing_by_name = {p.name: p for p in attempts}
            attempt_path = existing_by_name.get(attempt) or direct
            index = attempt_path / "index.html"
        return _read(index) or (
            f"<p>No <code>index.html</code> at <code>{index}</code>.</p>"
            "<p>A Cursor packet is not a previewable page until the frontier output "
            "has been written into that attempt folder.</p>"
        )

    def rate_ui(trial_id, attempt, visual, fit, copy, responsive, interaction, merge, cleanup, winner, notes):
        ns = argparse.Namespace(
            trial_id=trial_id.strip(),
            attempt=attempt.strip(),
            visual_quality=int(visual),
            game_fit=int(fit),
            copy_quality=int(copy),
            responsiveness=int(responsive),
            interaction_quality=int(interaction),
            mergeability=int(merge),
            cleanup_minutes=int(cleanup),
            winner=bool(winner),
            notes=notes or "",
        )
        return f"Saved `{rate_attempt(ns)}`"

    with gr.Blocks(title="Fallen Empire Model Arena", css=ARENA_CSS) as demo:
        gr.Markdown(
            "# Fallen Empire Model Arena\n"
            "One prompt, two competing builders. Code appears in local/frontier panes, previews render side by side, and grading saves the training record."
        )
        with gr.Group(elem_classes=["fe-shell"]):
            i_trial = gr.State("")
            i_local_attempt = gr.State("local_300")
            i_frontier_attempt = gr.State("frontier_api")
            i_prompt = gr.Textbox(label="Build Prompt", value=DEFAULT_TASK, lines=4)
            with gr.Row():
                i_run = gr.Button("Run Local + Frontier", variant="primary")
                i_preview_btn = gr.Button("Preview Current Code", variant="primary")
                i_save = gr.Button("Save Grades + Training Data", variant="primary")
            i_status = gr.Markdown()
            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    i_local_adapter = gr.Textbox(label="Local adapter", value=DEFAULT_LOCAL_ADAPTER)
                    i_frontier_model = gr.Textbox(label="Frontier model", value=os.environ.get("FRONTIER_MODEL", ""))
                    i_frontier_base = gr.Textbox(label="API base URL", value=os.environ.get("FRONTIER_API_BASE_URL", "https://api.openai.com/v1"))
                with gr.Row():
                    i_use_api = gr.Checkbox(label="Use Frontier API", value=True)
                    i_tokens = gr.Number(label="Max tokens", value=DEFAULT_ARENA_MAX_TOKENS, precision=0)
                    i_temp = gr.Number(label="Temperature", value=0.0)
            with gr.Accordion("Paste Cursor/frontier output instead of API", open=False):
                gr.Markdown("Paste a response with fenced blocks such as ` ```html path=index.html `. It will replace the Frontier lane.")
                paste_text = gr.Textbox(label="Frontier pasted output", lines=10)
                paste_btn = gr.Button("Import Into Frontier Lane")
                paste_status = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Local Code")
                    i_l_html = gr.Textbox(label="index.html", lines=14)
                    i_l_css = gr.Textbox(label="styles.css", lines=8)
                    i_l_js = gr.Textbox(label="script.js", lines=5)
                with gr.Column():
                    gr.Markdown("## Frontier Code")
                    i_f_html = gr.Textbox(label="index.html", lines=14)
                    i_f_css = gr.Textbox(label="styles.css", lines=8)
                    i_f_js = gr.Textbox(label="script.js", lines=5)
            i_preview_status = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Local Preview")
                    i_local_preview = gr.HTML()
                with gr.Column():
                    gr.Markdown("## Frontier Preview")
                    i_frontier_preview = gr.HTML()
            gr.Markdown("## Grade")
            i_winner = gr.Radio(["local", "frontier", "tie", "neither"], value="tie", label="Winner")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Local Scores")
                    i_lv = gr.Slider(1, 5, value=3, step=1, label="Visual")
                    i_lf = gr.Slider(1, 5, value=3, step=1, label="Game fit")
                    i_lc = gr.Slider(1, 5, value=3, step=1, label="Copy")
                    i_lr = gr.Slider(1, 5, value=3, step=1, label="Responsive")
                    i_li = gr.Slider(1, 5, value=3, step=1, label="Interaction")
                    i_lm = gr.Slider(1, 5, value=3, step=1, label="Mergeability")
                    i_ln = gr.Textbox(label="Local notes", lines=3)
                with gr.Column():
                    gr.Markdown("### Frontier Scores")
                    i_fv = gr.Slider(1, 5, value=3, step=1, label="Visual")
                    i_ff = gr.Slider(1, 5, value=3, step=1, label="Game fit")
                    i_fc = gr.Slider(1, 5, value=3, step=1, label="Copy")
                    i_fr = gr.Slider(1, 5, value=3, step=1, label="Responsive")
                    i_fi = gr.Slider(1, 5, value=3, step=1, label="Interaction")
                    i_fm = gr.Slider(1, 5, value=3, step=1, label="Mergeability")
                    i_fn = gr.Textbox(label="Frontier notes", lines=3)
            i_training_value = gr.Textbox(
                label="Training note",
                lines=3,
                placeholder="What should future training learn from this comparison?",
            )
            i_save_out = gr.Markdown()

        i_run.click(
            ide_run_split_ui,
            inputs=[i_prompt, i_local_adapter, i_frontier_model, i_frontier_base, i_use_api, i_tokens, i_temp],
            outputs=[
                i_trial,
                i_local_attempt,
                i_frontier_attempt,
                i_status,
                i_l_html,
                i_l_css,
                i_l_js,
                i_f_html,
                i_f_css,
                i_f_js,
                i_local_preview,
                i_frontier_preview,
            ],
        )
        i_preview_btn.click(
            ide_preview_ui,
            inputs=[
                i_trial,
                i_local_attempt,
                i_frontier_attempt,
                i_l_html,
                i_l_css,
                i_l_js,
                i_f_html,
                i_f_css,
                i_f_js,
            ],
            outputs=[i_local_preview, i_frontier_preview, i_preview_status],
        )
        paste_btn.click(
            import_frontier_code_ui,
            inputs=[i_trial, i_frontier_attempt, paste_text],
            outputs=[paste_status, i_f_html, i_f_css, i_f_js, i_frontier_preview],
        )
        i_save.click(
            ide_save_training_ui,
            inputs=[
                i_trial,
                i_local_attempt,
                i_frontier_attempt,
                i_l_html,
                i_l_css,
                i_l_js,
                i_f_html,
                i_f_css,
                i_f_js,
                i_winner,
                i_lv,
                i_lf,
                i_lc,
                i_lr,
                i_li,
                i_lm,
                i_fv,
                i_ff,
                i_fc,
                i_fr,
                i_fi,
                i_fm,
                i_ln,
                i_fn,
                i_training_value,
            ],
            outputs=[i_save_out],
        )
    return demo


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fallen Empire landing-page human evaluation arena")
    sub = parser.add_subparsers(dest="command", required=True)

    p_brief = sub.add_parser("brief", help="Write/update the shared landing-page brief")
    p_brief.add_argument("--out", type=Path, default=DEFAULT_BRIEF)
    p_brief.add_argument("--overwrite", action="store_true")

    p_create = sub.add_parser("create", help="Create a new landing-page trial")
    p_create.add_argument("--task", default=DEFAULT_TASK)
    p_create.add_argument("--trial-id", default=None)
    p_create.add_argument("--brief", type=Path, default=DEFAULT_BRIEF)

    p_packet = sub.add_parser("cursor-packet", help="Generate a Cursor-ready frontier packet")
    p_packet.add_argument("--trial-id", required=True)
    p_packet.add_argument("--attempt", default="frontier_cursor")

    p_scaffold = sub.add_parser("scaffold", help="Create a static attempt folder")
    p_scaffold.add_argument("--trial-id", required=True)
    p_scaffold.add_argument("--attempt", required=True)

    p_local = sub.add_parser("local-attempt", help="Generate a static-site attempt with local MLX")
    p_local.add_argument("--trial-id", required=True)
    p_local.add_argument("--attempt", default="local_300")
    p_local.add_argument("--adapter-path", default=DEFAULT_LOCAL_ADAPTER)
    p_local.add_argument("--model", default=os.environ.get("MODEL", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"))
    p_local.add_argument("--max-tokens", type=int, default=DEFAULT_ARENA_MAX_TOKENS)
    p_local.add_argument("--temp", type=float, default=0.0)

    p_serve = sub.add_parser("serve", help="Serve an attempt folder with Python http.server")
    p_serve.add_argument("--trial-id", required=True)
    p_serve.add_argument("--attempt", required=True)
    p_serve.add_argument("--port", type=int, default=8090)

    p_rate = sub.add_parser("rate", help="Save a human rating for an attempt")
    p_rate.add_argument("--trial-id", required=True)
    p_rate.add_argument("--attempt", required=True)
    p_rate.add_argument("--visual-quality", type=int, required=True)
    p_rate.add_argument("--game-fit", type=int, default=3)
    p_rate.add_argument("--copy-quality", type=int, default=3)
    p_rate.add_argument("--responsiveness", type=int, default=3)
    p_rate.add_argument("--interaction-quality", type=int, default=3)
    p_rate.add_argument("--mergeability", type=int, default=3)
    p_rate.add_argument("--cleanup-minutes", type=int, default=0)
    p_rate.add_argument("--winner", action="store_true")
    p_rate.add_argument("--notes", default="")

    p_ui = sub.add_parser("ui", help="Launch the Gradio trial arena")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=7863)
    p_ui.add_argument("--share", action="store_true")

    sub.add_parser("list", help="List landing-page trials")

    args = parser.parse_args()
    if args.command == "brief":
        print(write_brief(args.out, overwrite=args.overwrite))
    elif args.command == "create":
        trial = create_trial(args.task, brief_path=args.brief, trial_id=args.trial_id)
        packet = cursor_packet(trial.trial_id)
        print(f"Created trial: {trial.trial_id}")
        print(f"Trial dir: {trial.trial_dir}")
        print(f"Cursor packet: {packet}")
    elif args.command == "cursor-packet":
        print(cursor_packet(args.trial_id, args.attempt))
    elif args.command == "scaffold":
        print(scaffold_attempt(args.trial_id, args.attempt))
    elif args.command == "local-attempt":
        print(local_attempt(args.trial_id, args.attempt, args.adapter_path, args.model, args.max_tokens, args.temp))
    elif args.command == "serve":
        serve_attempt(args.trial_id, args.attempt, args.port)
    elif args.command == "rate":
        print(rate_attempt(args))
    elif args.command == "ui":
        app = build_app()
        app.launch(server_name=args.host, server_port=args.port, share=args.share)
    elif args.command == "list":
        list_trials()


if __name__ == "__main__":
    main()
