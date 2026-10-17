---
id: "0004"
title: "Pickling a linked Parameterized child round-trips without parent-graph cycles"
status: "In Progress"
priority: "High"
created: "2026-10-17"
last_updated: "2026-10-17"
related_tenets:
  - "trees-not-graphs-for-pickle"
stakeholders:
  - "GPy users"
tags:
  - pickle
---

# REQ-0004: Pickling a linked Parameterized child round-trips without parent-graph cycles

## Description

Pickling a `Parameterized` object that already has a parent must not serialize the whole upward graph or fail on incomplete-object cycles. Durable on-disk save remains `param_array` + reconstruct script; pickle must at least be safe for linked children.

## Acceptance Criteria

- [x] Regression test for linked-child pickle round-trip exists on the fix branch.
- [ ] PR #50 merges with green CI (or documented waiver for unrelated failures).
- [ ] GPy #605 reproduction passes against a released / tagged paramz including the fix.

## References

- PR: https://github.com/sods/paramz/pull/50
- GPy: https://github.com/SheffieldML/GPy/issues/605
- Related Tenets: trees-not-graphs-for-pickle
