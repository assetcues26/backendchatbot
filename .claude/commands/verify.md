---
description: Run the full quality gate (lint, types, tests) and report failures
---

Run all three, in this order, and report only what fails:

1. `.venv/Scripts/python.exe -m ruff check .`
2. `.venv/Scripts/python.exe -m mypy app`
3. `.venv/Scripts/python.exe -m pytest tests/ -q`

If the security tests skipped, say so explicitly and note that they need
`DATABASE_URL` pointed at a Postgres with pgvector — a green run that skipped
the leakage suite has not verified the access model.
