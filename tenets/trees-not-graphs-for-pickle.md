---
id: "trees-not-graphs-for-pickle"
title: "Pickle the parameter tree, not the whole parent graph"
status: "Active"
created: "2026-10-17"
last_reviewed: "2026-10-17"
review_frequency: "Annual"
conflicts_with: []
tags:
  - tenet
  - pickle
  - serialization
---

# Pickle the parameter tree, not the whole parent graph

## Tenet

**Description**: Pickling a `Parameterized` child that is already linked into a parent must not attempt to serialize the entire upward graph. Prefer durable save via `param_array` plus reconstruct script; when pickle is used, round-trips of linked children must stay cycle-safe.

**Quote**: *"Serialize the subtree you asked for."*

**Examples**:
- Omitting `_parent_` from the pickle memento (GPy #605 / paramz PR #50).
- README guidance: save `param_array`, rebuild the model, assign, initialize.

**Counter-examples**:
- Pickling a kernel attached to a GP and expecting a full live GP graph to resurrect cleanly.
- Relying on pickle as a cross-version on-disk format.

**Conflicts**:
- User expectation that "pickle anything" works like a database
- Resolution: document intended scope; fix cycle bugs; keep `param_array` as the durable path
