---
id: "2026-10-17_cip0004-epydoc-pages-disposition"
title: "Retire or demote epydoc GitHub Pages (docs/)"
status: "Proposed"
priority: "Medium"
created: "2026-10-17"
last_updated: "2026-10-17"
category: "documentation"
related_cips: ["0004"]
owner: "unassigned"
dependencies:
  - "2026-10-17_cip0004-sphinx-narrative-and-rtd"
tags:
  - backlog
  - documentation
  - github-pages
---

# Task: Epydoc Pages disposition

## Description

Stop serving unchecked-in-sync epydoc as current API docs. Prefer: Pages builds Sphinx or Pages disabled with RTD-only; minimum: historical stub/redirect (CIP-0004).

## Acceptance Criteria

- [ ] `docs/` / opendsi.cc paramz path does not claim to be current API docs without label
- [ ] Disposition recorded in CIP-0004 implementation status / CHANGELOG
- [ ] REQ-0007 legacy-site criterion satisfied

## Related

- CIP: 0004
- Requirements: REQ-0007
