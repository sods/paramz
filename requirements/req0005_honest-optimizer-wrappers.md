---
id: "0005"
title: "Optimizer wrappers expose upstream options honestly"
status: "Validated"
priority: "Medium"
created: "2026-10-17"
last_updated: "2026-10-17"
related_tenets:
  - "docs-match-behaviour"
stakeholders:
  - "GPy / paramz users"
tags:
  - optimization
---

# REQ-0005: Optimizer wrappers expose upstream options honestly

## Description

Thin wrappers around SciPy / climin should pass through options the upstream solver actually supports, and should not warn that supported options are ignored. Optional gradient callables needed for stochastic use should be injectable where the wrapper already owns the opt loop.

## Acceptance Criteria

- [x] L-BFGS-B `ftol` is accepted when SciPy supports it (#47).
- [x] Adadelta can take a caller-supplied gradient function (#24), or the PR is closed with rationale.
- [x] Verbose optimization IPython/notebook detection works or is removed (#31).

## References

- Issues: #47, #31
- PR: #24
- Related Tenets: docs-match-behaviour
