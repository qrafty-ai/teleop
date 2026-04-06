# Spec: Hand Robot Interface

## ADDED Requirements

### Requirement: BaseHandRobot abstract class
The system SHALL provide a `BaseHandRobot` abstract class extending `BaseRobot` for hand-specific functionality.

#### Scenario: Class inheritance
- **WHEN** implementing a hand robot
- **THEN** the class SHALL inherit from `BaseHandRobot` which extends `BaseRobot`

#### Scenario: Required properties
- **WHEN** defining a hand robot
- **THEN** it SHALL implement `finger_joint_names` property returning a dict mapping finger names to joint name lists

### Requirement: Finger joint naming
Hand robots SHALL expose finger joint names in a structured dictionary format.

#### Scenario: Finger mapping format
- **WHEN** accessing `finger_joint_names`
- **THEN** the result SHALL be `dict[str, list[str]]` with keys like "thumb", "index", "middle", "ring"

#### Scenario: Joint name lists
- **WHEN** accessing joints for a finger
- **THEN** the list SHALL contain joint names in order from base to tip

### Requirement: Handedness property
Hand robots SHALL expose their supported handedness.

#### Scenario: Single hand support
- **WHEN** querying a hand robot's handedness
- **THEN** the `handedness` property SHALL return `"right"`, `"left"`, or `"both"`

#### Scenario: Right-hand only v1
- **WHEN** implementing the Allegro hand
- **THEN** `handedness` SHALL return `"right"` for v1

### Requirement: Hand DOF calculation
The system SHALL calculate total hand DOF from finger joint names.

#### Scenario: DOF calculation
- **WHEN** accessing `hand_dof` property
- **THEN** it SHALL return the sum of joints across all fingers

#### Scenario: Allegro hand DOF
- **WHEN** creating an Allegro hand robot
- **THEN** `hand_dof` SHALL return 16 (4 fingers × 4 joints)

### Requirement: Build hand costs method
Hand robots SHALL implement `build_hand_costs` for retargeting optimization.

#### Scenario: Cost function signature
- **WHEN** calling `build_hand_costs`
- **THEN** it SHALL accept `fingertip_positions: jax.Array`, `q_current: jax.Array | None` and return `list[Cost]`

#### Scenario: Cost composition
- **WHEN** building costs for retargeting
- **THEN** costs SHALL include position matching, joint limits, and regularization
