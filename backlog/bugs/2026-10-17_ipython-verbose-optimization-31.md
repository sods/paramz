---
id: "2026-10-17_ipython-verbose-optimization-31"
title: "Fix IPython detection in verbose_optimization (#31)"
status: "Completed"
priority: "Low"
created: "2026-10-17"
last_updated: "2026-10-17"
category: "bugs"
related_cips: []
owner: "unassigned"
dependencies: []
tags:
  - backlog
  - optimization
---

# Task: Fix IPython checking in verbose_optimization

## Description

https://github.com/sods/paramz/issues/31 — notebook detection in `verbose_optimization.py` does not work. Update detection for modern IPython/Jupyter or remove the special case if obsolete.

## Acceptance Criteria

- [x] Detection works in current Jupyter, or special-case removed with changelog note
- [x] Issue #31 closed

## Related

- Requirements: REQ-0005
- Issues: #31
