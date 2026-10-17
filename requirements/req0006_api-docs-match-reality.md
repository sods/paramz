---
id: "0006"
title: "Public docs describe APIs that exist and behave as stated"
status: "Proposed"
priority: "Medium"
created: "2026-10-17"
last_updated: "2026-10-17"
related_tenets:
  - "docs-match-behaviour"
stakeholders:
  - "paramz users"
tags:
  - documentation
---

# REQ-0006: Public docs describe APIs that exist and behave as stated

## Description

Docstrings and Sphinx pages must not advertise unimplemented methods. Constraint and tying behaviour should be documented with working examples, including what paramz does *not* provide (e.g. general nonlinear sum-to-one constraints).

## Acceptance Criteria

- [ ] `tie_to` is implemented or removed from Param docs (#34).
- [ ] Constraint introspection has a short documented example (`m.constraints`) (#29).
- [ ] Sum-to-one / simplex requests are answered in docs (supported pattern or explicit unsupported) (#35).
- [ ] Stale answered questions (#29, #30) are closed or turned into doc PRs.

## References

- Issues: #29, #30, #34, #35
- Related Tenets: docs-match-behaviour
