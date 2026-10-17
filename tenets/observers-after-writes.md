---
id: "observers-after-writes"
title: "Meaningful writes notify the model"
status: "Active"
created: "2026-10-17"
last_reviewed: "2026-10-17"
review_frequency: "Annual"
conflicts_with:
  - "one-copy-in-memory"
tags:
  - tenet
  - observers
  - caching
---

# Meaningful writes notify the model

## Tenet

**Description**: After a user-facing parameter write, observers and caches must update so printed values, gradients, and predictions agree. Batch or internal buffer writes may defer notification, but the recommended public load/copy path must trigger `_trigger_params_changed()` (or equivalent).

**Quote**: *"If the numbers changed for the user, the model must know."*

**Examples**:
- `m[:] = loaded` notifies and refreshes derived state.
- `update_model(False)` batches algebra during construction, then a single notify at the end.
- Docs warn when a low-level buffer write does not notify.

**Counter-examples**:
- Documenting `m.param_array[:] = …` as the save/load API without saying observers stay stale.
- Auto-notifying on every internal buffer poke during `randomize` or gradient checks (defeats batching).

**Conflicts**:
- one-copy-in-memory (raw buffer writes are fast and silent)
- Resolution: keep silent buffer writes for internal batching; make the public reload path notifying (`m[:]` or explicit trigger)
