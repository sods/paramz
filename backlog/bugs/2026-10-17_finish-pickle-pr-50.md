---
id: "2026-10-17_finish-pickle-pr-50"
title: "Finish and merge pickle linked-child PR #50"
status: "In Progress"
priority: "High"
created: "2026-10-17"
last_updated: "2026-10-17"
category: "bugs"
related_cips: ["0002"]
owner: "Neil Lawrence"
dependencies: []
tags:
  - backlog
  - pickle
  - ci
---

# Task: Finish pickle PR #50

## Description

Land https://github.com/sods/paramz/pull/50. CI failed on `TestModel.test_likelihood_set` (likely unrelated); pickle regression passed. Fix or isolate the failure, merge, then confirm GPy #605.

## Acceptance Criteria

- [ ] CI green on the PR branch (or failure documented as pre-existing with waiver)
- [ ] PR #50 merged to master
- [ ] GPy #605 reproduction checked against the branch

## Related

- CIP: 0002
- PRs: #50
- Issues: GPy #605, #932

## Progress Updates

### 2026-10-17

Task created from open-PR survey. Branch checked out locally as `fix/605-pickle-used-parameterized`.
