---
id: "2026-10-17_finish-pickle-pr-50"
title: "Finish and merge pickle linked-child PR #50"
status: "Completed"
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

- [x] CI green on the PR branch (or failure documented as pre-existing with waiver)
- [x] PR #50 merged to master
- [x] GPy #605 reproduction checked against the branch

## Related

- CIP: 0002
- PRs: #50
- Issues: GPy #605, #932

## Progress Updates

### 2026-10-17

Task created from open-PR survey. Branch checked out locally as `fix/605-pickle-used-parameterized`.

### 2026-10-17 (later)

PR #50 merged. Reproduced GPy #605 against `paramz==0.10.1` (local master) + `GPy==1.14.2` (`SheffieldML/GPy`): pickle of `Matern52` after `GPRegression` attach loads with `_name` intact; restored kernel is detached (`has_parent() is False`).
