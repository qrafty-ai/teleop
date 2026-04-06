# docstring-improvement Specification

## Purpose
TBD - created by archiving change add-docstring-to-event-settings. Update Purpose after archive.
## Requirements
### Requirement: Event settings method has documentation

The `CommonCLI.event_settings()` method MUST have a Google-style docstring describing its purpose and return value.

#### Scenario: Docstring is present and complete

- **WHEN** examining the `event_settings()` method in `teleop_xr/common_cli.py`
- **THEN** a docstring SHALL be present
- **AND** the docstring SHALL describe the method's purpose
- **AND** the docstring SHALL document the return type (`EventSettings`)
- **AND** the docstring SHALL explain that it uses CLI timing configuration values
