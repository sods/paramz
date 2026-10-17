---
id: "2026-10-17_cip0004-fix-readme-sphinx-instructions"
title: "Fix README Sphinx build instructions and primary docs links"
status: "Proposed"
priority: "Medium"
created: "2026-10-17"
last_updated: "2026-10-17"
category: "documentation"
related_cips: ["0004"]
owner: "unassigned"
dependencies: []
tags:
  - backlog
  - documentation
  - sphinx
---

# Task: Fix README documentation section

## Description

Replace the GPy `sphinx-apidoc` path, state that Sphinx/RTD is the primary API docs home, and stop implying epydoc Pages is current (CIP-0004).

## Acceptance Criteria

- [ ] README build steps use `../paramz` (or document conf.py autodoc flow)
- [ ] Primary link is the chosen hosted Sphinx site
- [ ] Legacy Pages not advertised as current without a historical label

## Related

- CIP: 0004
- Requirements: REQ-0007
