# Game Arena Apply Contract

This contract defines the required assistant output shape for `scripts/game_task_arena.py` so model answers become applyable edits instead of `no_applyable_changes`.

## Output contract (strict)

- Return **exactly one** applyable artifact:
  - a unified diff that starts with `diff --git ...`, **or**
  - fenced full-file blocks with repo-relative paths.
- Do **not** emit prose, summaries, shell commands, analysis, or dual formats.
- For small UI edits, prefer fenced full-file blocks.
- Only write files inside task `allowed_paths`.

## Accepted fenced examples

```tsx path=src/components/ui/GameHUD.tsx
// full file body
```

```typescript file: src/lib/simStateSerialization.ts
// full file body
```

```path=src/app/test-env/[envId]/page.tsx
// full file body
```

You may also place a path marker in one fence and content in the next fence:

```text
src/components/ui/GameLoadingScreen.tsx
```

```tsx
// full file body
```

## Unified diff requirements

- Include proper headers (`diff --git`, `---`, `+++`, `@@` hunks).
- Diff must apply cleanly with `git apply --check`.
- Modify only allowed paths.

## Common failure causes to avoid

- Markdown explanation only; no diff and no writable fence.
- Fenced code without a parseable path.
- Paths outside allowed globs.
- Stub or partial declarations that drop existing exports.
- Renaming/removing exported symbols unless task explicitly asks for it.

## Export and TypeScript safety

- Preserve existing exported names from edited files unless explicitly requested.
- Keep TypeScript compileable after edits (`npx tsc --noEmit` in arena gate).
- Prefer minimal localized edits over broad rewrites.

