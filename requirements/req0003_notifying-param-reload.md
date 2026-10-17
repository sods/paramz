---
id: "0003"
title: "Recommended parameter reload path notifies observers"
status: "Proposed"
priority: "High"
created: "2026-10-17"
last_updated: "2026-10-17"
related_tenets:
  - "observers-after-writes"
  - "one-copy-in-memory"
stakeholders:
  - "GPy users"
  - "paramz maintainers"
tags:
  - observers
  - serialization
---

# REQ-0003: Recommended parameter reload path notifies observers

## Description

Users who restore parameters after save/load must get consistent views, caches, and predictions. The documented recommended API must either notify observers or clearly state that a low-level buffer write does not.

**Why this matters**: `param_array[:] =` is widely cargo-culted and silently leaves models inconsistent (#51, GPy #1217).

**Who benefits**: Anyone following README / GPy save-load recipes.

## Acceptance Criteria

- [ ] Docstrings and README recommend a notifying path (`m[:] = …` or equivalent).
- [ ] A regression test covers reload → predictions/views match.
- [ ] If auto-notify on `param_array.__setitem__` is added, batching via `update_model(False)` still works.

## References

- Issue: https://github.com/sods/paramz/issues/51
- GPy: https://github.com/SheffieldML/GPy/issues/1217
- Related Tenets: observers-after-writes; one-copy-in-memory
