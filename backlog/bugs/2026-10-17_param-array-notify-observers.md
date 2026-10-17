---
id: "2026-10-17_param-array-notify-observers"
title: "param_array[:] does not notify observers (#51)"
status: "Completed"
priority: "High"
created: "2026-10-17"
last_updated: "2026-10-17"
category: "bugs"
related_cips: ["0003"]
owner: "Neil Lawrence"
dependencies: []
tags:
  - backlog
  - observers
---

# Task: Fix or document param_array notify gap

## Description

https://github.com/sods/paramz/issues/51 — in-place `param_array` writes skip `_trigger_params_changed()`. Implement CIP-0003 (docs first, optional notifying wrapper).

## Acceptance Criteria

- [x] Docs/README recommend notifying reload path
- [x] Regression test added
- [x] Issue #51 closed or updated with disposition

## Related

- CIP: 0003
- Issues: #51
- Downstream: GPy #1217, GPy PR #1221
