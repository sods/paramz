---
id: "2026-10-17_tie-to-docs-or-implement-34"
title: "Resolve undocumented/missing tie_to (#34)"
status: "Completed"
priority: "Medium"
created: "2026-10-17"
last_updated: "2026-10-17"
category: "documentation"
related_cips: []
owner: "unassigned"
dependencies: []
tags:
  - backlog
  - documentation
  - api
---

# Task: tie_to docs vs implementation

## Description

https://github.com/sods/paramz/issues/34 — Param docs mention `tie_to`, but it was never implemented. Either implement tying, or remove/clarify the docstring and document the `parameters_changed` sync pattern users already use.

## Acceptance Criteria

- [x] Docstring no longer promises a missing API, or `tie_to` exists with tests
- [x] Short example of sharing values across params documented
- [x] Issue #34 closed

## Related

- Requirements: REQ-0006
- Issues: #34
