## Why

The `CommonCLI.event_settings()` method is a public API that creates `EventSettings` objects for button event detection, but it lacks documentation. This makes it unclear what the method does and what it returns, especially since it has a lazy import pattern. Adding a proper docstring will improve code maintainability and help developers understand the codebase faster.

## What Changes

- Add a Google-style docstring to `CommonCLI.event_settings()` method in `teleop_xr/common_cli.py`
- Include description of the method's purpose
- Document the return value with type information

## Capabilities

### New Capabilities
<!-- This is a documentation improvement, not a new capability with spec-level requirements -->

### Modified Capabilities
<!-- No existing spec modifications needed -->

## Impact

- `teleop_xr/common_cli.py`: Add docstring to `event_settings()` method (lines 13-19)
- No breaking changes
- No API changes
- Improves documentation coverage
