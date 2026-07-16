# ThermoReconLab Agent Guidelines

## Scope and environment

- Work only inside the currently opened ThermoReconLab repository.
- Use the existing `.venv`. In Windows PowerShell, invoke Python with `& ".\.venv\Scripts\python.exe"`.
- Inspect the exact current files before editing; never reconstruct them from assumptions.
- Modify only files directly required by the current task, and do not begin another phase or task without explicit user instruction.

## Repository safety

- Preserve all existing modified and untracked work.
- Never reset, restore, checkout, clean, stash, delete, overwrite, or otherwise discard existing work.
- Do not run destructive Git commands, commit, or push.
- Do not modify generated files under `outputs/`, `presentations/`, `**/__pycache__/`, or `.pytest_cache/`.

## Implementation rules

- Preserve existing public APIs and passing tests.
- Keep implementations deterministic and validate public inputs.
- Do not add machine-learning or AI reconstruction unless explicitly requested.
- Do not change the original `reconstruct_tikhonov()` solver. The separate `reconstruct_smooth_tikhonov()` solver already exists.
- Never claim scientific improvement from appearance alone; compare numerical metrics.

## Validation after implementation tasks

Run, in order:

1. Targeted tests for the changed behavior.
2. The full suite: `& ".\.venv\Scripts\python.exe" -m pytest -q`.
3. `git diff --check`.

## Current project state

- Phase 1 is complete.
- Phase 2 is paused at 3/5 tasks (60%): repeated-noise study and plot, sensor-layout study and plot, and fair reconstruction comparison with shared source scaling are complete.
- The latest verified full suite has 356 passing tests.
- The smooth solver improved numerical source errors and removed negative values, but produced broad connected hotspots.
- The algorithm side task is paused before compact nonnegative reconstruction; do not resume it without explicit instruction.
