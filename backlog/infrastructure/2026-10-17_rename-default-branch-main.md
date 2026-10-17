---
id: "2026-10-17_rename-default-branch-main"
title: "Rename default branch master → main"
status: "Completed"
priority: "Low"
created: "2026-10-17"
last_updated: "2026-10-17"
category: "infrastructure"
related_cips: []
owner: "Neil Lawrence"
dependencies: []
tags:
  - backlog
  - infrastructure
  - github
---

# Task: Rename default branch to main

## Description

Rename GitHub default branch `master` → `main`. Tracked as issue #55.

## Acceptance Criteria

- [x] Default branch renamed on GitHub
- [x] CI workflow + badges updated
- [x] Open PRs retargeted (none remaining at rename time)
- [x] Issue #55 closed

## Related

- Issues: #55

## Progress Updates

### 2026-10-17

GitHub default branch renamed to `main`. Updated Actions trigger, README/codecov badges (Travis badge replaced with Actions), and Travis deploy branch filter. Local `main` tracks `origin/main`; stale `master` remote pruned.
