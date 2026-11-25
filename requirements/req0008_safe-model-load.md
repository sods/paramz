---
id: "0008"
title: "Loading a model file does not execute untrusted code by default"
status: "Implemented"
priority: "High"
created: "2026-11-22"
last_updated: "2026-11-22"
related_tenets: []
stakeholders:
  - "paramz users"
  - "GPy users"
  - "maintainers"
tags:
  - security
  - pickle
  - cwe-502
---

# REQ-0008: Loading a model file does not execute untrusted code by default

## Description

`paramz.load` (and callers such as `GPy.load`) must not run arbitrary code from a crafted model file under the default load path. Users who load a received or published model should get either a restored object or a clear load failure — not side effects from builtins gadgets such as `exec`.

An explicit escape hatch for unrestricted pickle may exist for trusted archives, but it must be opt-in and documented as unsafe.

**Why this matters**: Model files are shared between collaborators and published with papers. Stock `pickle.load` makes those files executable under the loader's privileges (CWE-502; GitHub issue #60).

**Who benefits**: Anyone loading paramz/GPy pickles from disk or third parties.

## Acceptance Criteria

- [x] Default `paramz.load` rejects a crafted pickle whose reconstruction uses disallowed callables (e.g. `builtins.exec`).
- [x] Legitimate paramz round-trips (and typical GPy model loads when exercised) still succeed under the default path.
- [x] Any unrestricted pickle path is opt-in, named clearly, and documented as trusting the file author.
- [x] User-facing docs state that model pickles should only be loaded from trusted sources, even with the restricted loader.
- [ ] Patch release notes credit the reporter and reference the issue / advisory.

## References

- GitHub: https://github.com/sods/paramz/issues/60
- CWE-502: Deserialization of Untrusted Data
