# Bug Fix Common Errors (Arena/Routing)

Use this checklist when reviewing model-generated patches for game-task arena flows.

## High Priority Failure Patterns

1. No applyable patch output

- Missing unified diff header (`diff --git ...`) when output is intended as a diff.
- Fenced code block has no repo-relative path.
- Path marker is present but outside allowed globs.
- Output includes prose instead of patch/file content.

1. TypeScript compile failures (`tsc`)

- Introduced symbolets tesls are not imported.
- Imported symbols do not exist in target module.
- Referenced store/type fields do not exist on real game types.
- JSX/TSX typing mismatch (`.ts` vs `.tsx` component path confusion).
- Inferred schema drift: patch invents fields not present in `src/types/game.ts`.

1. Export regression

- Existing named export removed from touched files.
- Default export changed unexpectedly in shared UI/component modules.
- Helper function renamed without updating import sites.

1. Wrong file targeting

- Patch edits similarly named but non-rendered files.
- Context mentions `/test-env/...` route but patch ignores route-owned components.
- Changes land in dead/unused files while visible surface remains unchanged.

1. Scope violations

- Writes outside `allowed_paths` for task.
- Broad refactors for a small UI/economy/combat task.
- Introduces unrelated architectural changes in a task-specific patch.

## Review Hints For Bug-Fix Loop

- Verify output shape first: "can this be applied deterministically?"
- Prefer minimal edits over rewrites unless parse/apply fails.
- Preserve existing exported identifiers unless task explicitly asks for rename.
- For UI tasks, ensure changed files are those rendered by the preview route.
- For economy/combat/HUD tasks, cross-check field names against `src/types/game.ts` and `src/store/useGameStore.ts`.

## Quick Triage Tags

- `parse_fail`: malformed diff/fence/path extraction failure.
- `apply_fail`: patch does not apply cleanly.
- `no_applyable_changes`: output has no usable patch/file block.
- `tsc_fail`: compile/type error.
- `export_fail`: exported symbol dropped from touched file.
- `schema_hallucination`: invented game/store/type fields.
- `wrong_target_file`: edits non-rendered or irrelevant files.