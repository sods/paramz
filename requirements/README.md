# VibeSafe Requirements

This directory contains requirements for VibeSafe itself. Requirements define **WHAT** needs to be built.

## What vs How vs Do

VibeSafe uses a clear hierarchy to separate concerns:

| Level | Purpose | Question | Component | Example |
|-------|---------|----------|-----------|---------|
| **WHY** | Foundation principles | Why does this matter? | Tenets | "simplicity-of-use" |
| **WHAT** | Desired outcomes | What should be true? | Requirements | "One-line installation" |
| **HOW** | Design approach | How will we achieve it? | CIPs | "Minimal install script with auto-detection" |
| **DO** | Execution tasks | What are we doing now? | Backlog | "Write install-minimal.sh tests" |

### WHAT: Requirements (This Directory)

Requirements describe **outcomes and desired states**, not implementation details:

✅ **Good Requirements (WHAT)**:
- "Users can install VibeSafe with a single command"
- "Project tenets are automatically available to AI assistants"
- "Documentation stays synchronized with implementation"
- "System files don't clutter user repositories"

❌ **Bad Requirements (HOW in disguise)**:
- "Create install-minimal.sh script" ← This is implementation (belongs in CIP/Backlog)
- "Use PyYAML for parsing" ← This is design decision (belongs in CIP)
- "Add --no-color flag to whats-next" ← This is a specific task (belongs in Backlog)

### Decision Guide: Am I Writing WHAT or HOW?

Ask yourself:
1. **Does this describe an outcome or a method?**
   - Outcome → Requirement (WHAT)
   - Method → CIP (HOW)

2. **Could multiple approaches achieve this?**
   - Yes → Requirement (WHAT)
   - No, it's specific → CIP or Backlog (HOW/DO)

3. **Does it start with a verb describing work?**
   - "Create...", "Implement...", "Add..." → Usually HOW/DO
   - "Users can...", "System should...", "X must be..." → Usually WHAT

### The Flow

```
Tenet (WHY) ──informs──> Requirement (WHAT) ──guides──> CIP (HOW) ──breaks into──> Backlog (DO) ──produces──> Implementation
```

**Example Chain**:
1. **Tenet (WHY)**: "Simplicity at All Levels" - minimize friction for users
2. **Requirement (WHAT)**: REQ-007 "Automatic System Updates" - users shouldn't manually maintain VibeSafe files
3. **CIP (HOW)**: CIP-000E "Clean Installation Philosophy" - every install is a reinstall, overwrite system files
4. **Backlog (DO)**: Task "Implement clean installation in install-minimal.sh" - specific coding work
5. **Implementation**: The actual script with logic

## Format

Each requirement uses YAML frontmatter for metadata and a simple structure:

```yaml
---
id: "XXXX"
title: "Requirement Title"
status: "Proposed"  # Proposed, Ready, In Progress, Implemented, Validated, Deferred, Rejected
priority: "Medium"  # High, Medium, Low
created: "YYYY-MM-DD"
last_updated: "YYYY-MM-DD"
related_tenets: []  # Bottom-up: Which tenets (WHY) inform this requirement?
stakeholders: []  # Optional: Who cares about this requirement?
tags: []  # Optional: Categorization tags
---

# REQ-XXXX: Requirement Title

## Description
Brief description of what needs to be built (2-3 paragraphs max)

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Notes (Optional)
Any additional context
```

## Status Values

- **Proposed**: Initial requirement, needs refinement
- **Ready**: Fully defined, ready for implementation
- **In Progress**: Currently being implemented
- **Implemented**: Code complete, needs validation
- **Validated**: Implementation verified against acceptance criteria

## File Naming

Requirements are named: `reqXXXX_short-description.md` (4-digit hexadecimal, lowercase)

Examples:
- `req0001_yaml-standardization.md`
- `req0002_simplify-requirements.md`
- `req000A_minimal-version-control.md` (hex: A = 10)
- `req00FF_future-requirement.md` (hex: FF = 255)

## Integration

Requirements connect tenets (WHY) to implementation (HOW/DO):

```
Tenets (WHY) → Requirements (WHAT) → CIP (HOW) → Backlog Task (DO) → Implementation
```

**Linking Structure** (bottom-up):
- Requirements reference `related_tenets` (WHY informs WHAT)
- CIPs reference `related_requirements` (WHAT informs HOW)
- Backlog tasks reference `related_cips` (HOW informs DO)
- Scripts query inverse: "Which requirements relate to this tenet?"

This creates bidirectional traceability while keeping each component focused on its immediate context.

## Need Help?

VibeSafe provides thinking tools for requirements discovery in the main documentation. These are reference materials to consult when stuck, not mandatory process steps.

