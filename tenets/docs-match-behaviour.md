---
id: "docs-match-behaviour"
title: "Documented APIs must exist and match behaviour"
status: "Active"
created: "2026-10-17"
last_reviewed: "2026-10-17"
review_frequency: "Annual"
conflicts_with: []
tags:
  - tenet
  - documentation
---

# Documented APIs must exist and match behaviour

## Tenet

**Description**: Sphinx / docstring API that users can call must be implemented, or clearly marked unimplemented. Warnings about optimizer options and constraint APIs should match the underlying SciPy / transformation behaviour.

**Quote**: *"If it is in the docs, it is real — or labelled missing."*

**Examples**:
- Remove or implement `tie_to` rather than leaving a ghost method in Param docs (#34).
- L-BFGS-B `ftol` warning should reflect SciPy's actual support (#47).
- Constraint introspection docs should point at `m.constraints` with a worked example (#29).

**Counter-examples**:
- Shipping docstrings for methods that raise `AttributeError`.
- Leaving incorrect "option ignored" warnings after upstream SciPy changes.
