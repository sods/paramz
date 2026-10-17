---
id: "0007"
title: "Published user documentation is trustworthy and has one primary home"
status: "Proposed"
priority: "Medium"
created: "2026-10-17"
last_updated: "2026-10-17"
related_tenets:
  - "docs-match-behaviour"
stakeholders:
  - "paramz users"
  - "GPy users"
  - "maintainers"
tags:
  - documentation
  - sphinx
  - readthedocs
---

# REQ-0007: Published user documentation is trustworthy and has one primary home

## Description

Users who follow README links or land on hosted API docs should see guidance that matches current paramz behaviour (including NumPy 2 shared-buffer semantics, pickle of linked children, and notifying reload via `m[:]`). There should be one primary published API documentation home; secondary or legacy sites must not contradict it without a clear redirect or deprecation notice.

**Why this matters**: Dual Sphinx/RTD and epydoc GitHub Pages sites, plus a README that still mentions GPy paths, make it easy to trust stale material after the 0.10.x fixes.

**Who benefits**: Anyone integrating paramz (especially via GPy) and maintainers closing CIP compression loops.

## Acceptance Criteria

- [ ] README documentation section describes how to build Sphinx from `doc/` against `paramz` (not GPy) and links to the chosen primary hosted docs.
- [ ] The primary hosted API docs rebuild from current source on the default branch.
- [ ] Closed CIP-0002 and CIP-0003 outcomes appear in formal user-facing docs (not only CHANGELOG / vibesafe), and both CIPs have `compressed: true`.
- [ ] Legacy epydoc Pages (`docs/`) is either retired, redirected, or explicitly labelled historical and not linked as current API docs.

## References

- **Related Tenets**: docs-match-behaviour
- **Related CIPs**: CIP-0004
- External: https://paramz.readthedocs.io/ · http://opendsi.cc/paramz/
