---
id: "2026-10-17_verify-close-numpy-data-issue-17"
title: "Verify NumPy .data fix and close #17"
status: "Ready"
priority: "High"
created: "2026-10-17"
last_updated: "2026-10-17"
category: "bugs"
related_cips: []
owner: "unassigned"
dependencies: []
tags:
  - backlog
  - numpy
---

# Task: Verify and close issue #17

## Description

https://github.com/sods/paramz/issues/17 tracked unsafe `ndarray.data` assignment. PR #49 (NumPy compatibility) merged 2026-07-29. Confirm master no longer assigns `.data`, run relevant tests, close #17 or file remaining gaps.

## Acceptance Criteria

- [ ] Confirm no `.data =` assignment remains on supported paths
- [ ] Comment on #17 with verification notes
- [ ] Close #17 or open follow-up for residual work

## Related

- Requirements: REQ-0002
- Issues: #17
- PRs: #49
