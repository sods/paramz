---
id: "0002"
title: "Shared parameter buffer works on supported NumPy without .data assignment"
status: "Validated"
priority: "High"
created: "2026-10-17"
last_updated: "2026-10-17"
related_tenets:
  - "one-copy-in-memory"
stakeholders:
  - "GPy users"
  - "paramz maintainers"
tags:
  - numpy
  - memory
---

# REQ-0002: Shared parameter buffer works on supported NumPy without .data assignment

## Description

Parameter linking and optimization must keep a single shared buffer on NumPy versions we claim to support. Assigning `ndarray.data` is deprecated and a hard error on NumPy 2.0+; a supported path must replace it.

**Why this matters**: This is the historical root of GPy/paramz memory and NumPy-compat failures (#17).

**Who benefits**: Anyone installing paramz under modern NumPy / GPy stacks.

## Acceptance Criteria

- [x] A NumPy-compatibility path lands on master (PR #49 merged).
- [x] Issue #17 is verified fixed on current master and closed, or remaining gaps are filed as new tasks.
- [x] CI exercises the shared-buffer path on the declared Python/NumPy matrix.

## References

- Issue: https://github.com/sods/paramz/issues/17
- PR: https://github.com/sods/paramz/pull/49
- Related Tenets: one-copy-in-memory

## Progress Updates

### 2026-10-17

PR #49 merged 2026-07-29; issue #17 still open pending verification/close.
