# Game Arena Standard Dev Patch Guide

Scope: `hud-status-summary` and `economy-tooltip` style tasks.

Goal: produce small, applyable, TypeScript-safe HUD/economy edits without broad rewrites.

## Hard requirements

- Make at least one concrete change in an allowed file (no noop output).
- Keep edits localized; avoid rewriting the full `GameHUD` file.
- Preserve existing exports and existing schema names.
- Keep imports minimal and valid; never emit giant duplicated import lists.
- Output only one applyable artifact shape (diff or fenced full-file blocks).

## Preferred edit strategy

1. Start from an existing HUD/economy render block.
2. Add a compact status/tooltip section with existing store fields.
3. Reuse existing style tokens/classes already present nearby.
4. If adding a helper component:
   - add all imports explicitly,
   - type selector params (no implicit `any`),
   - keep it in an existing file unless a new file is clearly required.

## Anti-patterns to avoid

- Massive top-of-file import rewrites.
- New component files that are not fully wired/imported.
- Broad constant dumps or invented schema symbols.
- Returning prose after an initial failed apply instead of corrected code.

## Quality gate before final output

- The patch is applyable.
- TypeScript compiles for touched code paths.
- Exports are preserved.
