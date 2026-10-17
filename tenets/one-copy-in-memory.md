---
id: "one-copy-in-memory"
title: "One copy of parameters in memory"
status: "Active"
created: "2026-10-17"
last_reviewed: "2026-10-17"
review_frequency: "Annual"
conflicts_with: []
tags:
  - tenet
  - memory
  - numpy
---

# One copy of parameters in memory

## Tenet

**Description**: Parameter trees share a single contiguous NumPy buffer. Views and child `Param` objects read and write that buffer; they do not each own a separate array. This is the core performance contract of paramz and must survive NumPy API changes.

**Quote**: *"One buffer, many views."*

**Examples**:
- Linking a parameter into a model remaps its storage onto the parent's `param_array`.
- Optimization steps write the flat array once; child views see the new values without Python loops.
- NumPy 2.x compatibility work preserves shared storage without assigning `.data`.

**Counter-examples**:
- Giving each `Param` its own independent array and syncing by copy each step.
- Replacing `param_array` with a new object instead of writing into the existing buffer.
- Silencing NumPy warnings without a replacement strategy when `.data` assignment breaks.

**Conflicts**:
- Convenience APIs that copy on every assignment
- Resolution: prefer in-place buffer writes; document when a copy is intentional
