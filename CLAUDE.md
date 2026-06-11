# CLAUDE.md

Agentic security-alert triage engine exposed over MCP. **Portfolio project** — favor
correctness, honest metrics, and defensible design over headline numbers. Architecture, key
seams, module layout, and design decisions live in `README.md`; don't duplicate them here.

## Environment

- Python 3.12. Use the project venv, not a bare `python`:
  - Windows: `.venv\Scripts\python.exe`
  - macOS/Linux: `.venv/bin/python`

## Verify (all offline, no API key, must stay clean — CI enforces them)

- Tests: `python -m pytest`
- Types: `python -m mypy`
- Lint + format: `python -m ruff check . && python -m ruff format --check .`

## Conventions

- TDD with frequent small commits; work on a feature branch and open a PR (PRs are squash-merged).
- Stage explicit paths when committing — never `git add -A` from the repo root.
- `before_interview.md` / `cv_info.md` are personal interview notes (gitignored) — not part of the
  project; leave them alone.
