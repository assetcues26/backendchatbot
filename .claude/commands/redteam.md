---
description: Run only the RBAC leakage and tampering suites
---

Run `.venv/Scripts/python.exe -m pytest tests/security -q -v`.

These are the tests that prove information cannot cross a role boundary. If
they skip for want of a database, say so plainly rather than reporting success.

For any failure, name which guardrail it belongs to (G1-G7, see
`docs/SECURITY.md`) and what a real user could have seen as a result.
