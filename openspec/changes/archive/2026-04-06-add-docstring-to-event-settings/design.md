## Context

The `CommonCLI` dataclass in `teleop_xr/common_cli.py` provides shared CLI configuration for the demo and ROS2 modules. The `event_settings()` method (lines 13-19) creates an `EventSettings` object using the CLI's timing configuration values, but it currently lacks a docstring.

Looking at the codebase conventions in `teleop_xr/events.py`, the project uses Google-style docstrings with proper descriptions and return value documentation.

## Goals / Non-Goals

**Goals:**
- Add a clear, Google-style docstring to `event_settings()` method
- Document the method's purpose and return value
- Follow existing codebase conventions

**Non-Goals:**
- No API changes
- No behavior changes
- No refactoring of existing code

## Decisions

### Decision 1: Docstring style

**Choice:** Use Google-style docstring format

**Rationale:** This matches the existing codebase conventions seen in `teleop_xr/events.py`. The style includes:
- Brief description of purpose
- `Returns:` section documenting the return type and what it contains

### Decision 2: Content scope

**Choice:** Focus on explaining the lazy import pattern and return value

**Rationale:** The method uses a lazy import (imports `EventSettings` inside the method rather than at module level), which is worth explaining. The docstring should clarify that this method instantiates `EventSettings` using the CLI's configured timing values.

## Risks / Trade-offs

No significant risks for this documentation-only change.
