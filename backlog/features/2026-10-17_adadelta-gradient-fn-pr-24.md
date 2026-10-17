---
id: "2026-10-17_adadelta-gradient-fn-pr-24"
title: "Decide merge or close Adadelta fp PR #24"
status: "Proposed"
priority: "Low"
created: "2026-10-17"
last_updated: "2026-10-17"
category: "features"
related_cips: []
owner: "unassigned"
dependencies: []
tags:
  - backlog
  - optimization
  - stale
---

# Task: Triage Adadelta gradient PR #24

## Description

https://github.com/sods/paramz/pull/24 (2017) adds `fp=` kwarg to Adadelta for stochastic gradients. Diff is +5/−1 and still mergeable. Rebase/test under current code, merge if still useful for GPy, otherwise close as stale.

## Acceptance Criteria

- [ ] Rebased or verified against master
- [ ] Merged with note, or closed with rationale
- [ ] REQ-0005 acceptance item updated

## Related

- Requirements: REQ-0005
- PRs: #24
