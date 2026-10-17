---
id: "0001"
title: "VibeSafe governance artifacts exist and stay in sync with tooling"
status: "Validated"
priority: "Medium"
created: "2026-10-17"
last_updated: "2026-10-17"
related_tenets:
  - "docs-match-behaviour"
stakeholders:
  - "Neil Lawrence"
tags:
  - vibesafe
  - governance
---

# REQ-0001: VibeSafe governance artifacts exist and stay in sync with tooling

## Description

Non-trivial structural and design work in paramz should be traceable through WHAT → HOW → DO (requirements → CIP → backlog). VibeSafe scaffolding may stay local/gitignored; project content (tenets, requirements, CIPs, tasks) is committed when it captures real decisions.

**Why this matters**: paramz is a small library with long-lived open issues; governance keeps AI-assisted and intermittent maintainer work aligned.

**Who benefits**: Maintainers and anyone reviewing design-heavy PRs.

## Acceptance Criteria

- [x] At least one requirement documents governance outcomes.
- [x] At least one CIP documents VibeSafe adoption.
- [x] Open GitHub issues/PRs are triaged into backlog (or explicitly deferred/closed).
- [x] `./whats-next` and the structure validator run cleanly against the working tree.
- [x] Governance content committed on a dedicated branch/PR.

## References

- Discussion: https://github.com/sods/paramz/discussions/52
- Related Tenets: docs-match-behaviour

## Progress Updates

### 2026-10-17

Requirement created after installing VibeSafe and surveying open issues/PRs.
