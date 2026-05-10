#!/usr/bin/env python3
"""Regression checks for Game Task Arena model-output parsing."""

from __future__ import annotations

import tempfile
from pathlib import Path

from game_task_arena import apply_fenced_files, parse_exported_identifiers


ALLOWED = ["src/**/*.tsx", "src/**/*.ts", "src/**/*.css"]


def run_case(name: str, text: str, expected_path: str, expected_first_line: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / expected_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("old\n", encoding="utf-8")
        ok, written = apply_fenced_files(root, text, ALLOWED, root / "apply.log")
        body = target.read_text(encoding="utf-8")
        first_line = body.splitlines()[0] if body.splitlines() else ""
        assert ok, f"{name}: parser did not write any file"
        assert written == [expected_path], f"{name}: wrote {written}, expected {[expected_path]}"
        assert first_line == expected_first_line, f"{name}: first line {first_line!r}, expected {expected_first_line!r}"
        assert not body.startswith(expected_path), f"{name}: wrote path marker as code"


def test_parse_exports() -> None:
    src = """
export async function Alpha() {}
export function Beta() {}
export const Gamma: number = 1;
export class Delta {}
export interface Epsilon {}
export enum Zeta { A }
export type Eta = string;
export type { Theta, Iota }
export { Kappa as Lambda, type Mu }
export default function NamedDefault() {}
"""
    names = parse_exported_identifiers(src)
    assert "Alpha" in names and "Beta" in names
    assert "Gamma" in names and "Delta" in names
    assert "Epsilon" in names and "Zeta" in names
    assert "Eta" in names
    assert "Theta" in names and "Iota" in names
    assert "Lambda" in names and "Mu" in names
    assert "NamedDefault" in names


def main() -> None:
    test_parse_exports()
    run_case(
        "path attribute",
        """```tsx path="src/components/ui/GameLoadingScreen.tsx"
'use client';
export function GameLoadingScreen() { return null; }
```""",
        "src/components/ui/GameLoadingScreen.tsx",
        "'use client';",
    )
    run_case(
        "bare info path",
        """```tsx src/components/ui/GameLoadingScreen.tsx
'use client';
export function GameLoadingScreen() { return null; }
```""",
        "src/components/ui/GameLoadingScreen.tsx",
        "'use client';",
    )
    run_case(
        "body comment path",
        """```tsx
// src/components/ui/GameLoadingScreen.tsx
'use client';
export function GameLoadingScreen() { return null; }
```""",
        "src/components/ui/GameLoadingScreen.tsx",
        "'use client';",
    )
    run_case(
        "body path label",
        """```tsx
// path: src/components/ui/GameLoadingScreen.tsx
'use client';
export function GameLoadingScreen() { return null; }
```""",
        "src/components/ui/GameLoadingScreen.tsx",
        "'use client';",
    )
    run_case(
        "separate marker fence",
        """```tsx
src/components/ui/GameLoadingScreen.tsx
```
```tsx
'use client';
export function GameLoadingScreen() { return null; }
```""",
        "src/components/ui/GameLoadingScreen.tsx",
        "'use client';",
    )
    run_case(
        "markdown heading path",
        """## `src/components/ui/GameLoadingScreen.tsx`

```tsx
'use client';
export function GameLoadingScreen() { return null; }
```""",
        "src/components/ui/GameLoadingScreen.tsx",
        "'use client';",
    )
    run_case(
        "ts resolves to existing tsx",
        """```tsx
// src/components/ui/GameLoadingScreen.ts
'use client';
export function GameLoadingScreen() { return null; }
```""",
        "src/components/ui/GameLoadingScreen.tsx",
        "'use client';",
    )
    print("parser regression cases passed")


if __name__ == "__main__":
    main()
