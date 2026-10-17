---
id: "2026-10-17_constraints-docs-and-stale-questions"
title: "Constraints docs; close or answer stale Q&A (#29 #30 #35)"
status: "Completed"
priority: "Low"
created: "2026-10-17"
last_updated: "2026-10-17"
category: "documentation"
related_cips: []
owner: "unassigned"
dependencies: []
tags:
  - backlog
  - documentation
  - constraints
---

# Task: Constraints documentation and stale questions

## Description

Triage documentation-shaped issues:

- #29 — how to read constraints (`m.constraints`); partially answered 2018, still open
- #30 — per-iteration objective during optimization; unanswered since 2019
- #35 — sum-to-one / simplex constraint; not a built-in; document softmax/transform pattern or explicit unsupported

## Acceptance Criteria

- [x] Doc section or FAQ covering constraint introspection
- [x] #30 answered (callback / verbose opt) or closed as out of scope
- [x] #35 answered with supported pattern or wontfix
- [x] Issues closed or labelled after response

## Related

- Requirements: REQ-0006
- Issues: #29, #30, #35
