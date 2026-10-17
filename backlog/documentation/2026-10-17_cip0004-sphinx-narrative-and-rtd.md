---
id: "2026-10-17_cip0004-sphinx-narrative-and-rtd"
title: "Sphinx narrative page + local build + RTD republish"
status: "Proposed"
priority: "Medium"
created: "2026-10-17"
last_updated: "2026-10-17"
category: "documentation"
related_cips: ["0004"]
owner: "unassigned"
dependencies:
  - "2026-10-17_cip0004-fix-readme-sphinx-instructions"
tags:
  - backlog
  - documentation
  - sphinx
  - readthedocs
---

# Task: Sphinx narrative and RTD republish

## Description

Repair `doc/` so Sphinx builds; add RST covering buffer model, pickle (CIP-0002), and notifying reload (CIP-0003); republish ReadTheDocs from `main`.

## Acceptance Criteria

- [ ] `make html` (or equivalent) succeeds locally
- [ ] Narrative page in toctree
- [ ] RTD build from `main` succeeds and shows new content
- [ ] CIP-0002 and CIP-0003 set `compressed: true`

## Related

- CIP: 0004
- Requirements: REQ-0007
