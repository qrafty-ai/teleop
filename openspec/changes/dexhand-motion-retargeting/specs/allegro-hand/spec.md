# Spec: Allegro Hand

## ADDED Requirements

### Requirement: Allegro Hand robot implementation
The system SHALL provide an `AllegroHandRobot` class implementing `BaseHandRobot`.

#### Scenario: Class definition
- **WHEN** importing the Allegro hand robot
- **THEN** `AllegroHandRobot` SHALL be available from `teleop_xr.ik.robots.allegro`

#### Scenario: Registration
- **WHEN** listing registered robots
- **THEN** `"allegro-hand"` SHALL appear as an entry point

### Requirement: URDF loading via RAM
The Allegro hand SHALL load its URDF from dexsuite/dex-urdf repository via RAM.

#### Scenario: URDF source
- **WHEN** loading the default URDF
- **THEN** it SHALL fetch from `https://github.com/dexsuite/dex-urdf.git`

#### Scenario: URDF path
- **WHEN** resolving the URDF file
- **THEN** it SHALL use path `robots/hands/allegro_hand/allegro_hand_right_glb.urdf`

#### Scenario: Mesh loading
- **WHEN** loading the URDF
- **THEN** GLB meshes SHALL be resolved from the repository root

### Requirement: Finger joint mapping
The Allegro hand SHALL map fingers to specific joint names.

#### Scenario: Thumb joints
- **WHEN** accessing `finger_joint_names["thumb"]`
- **THEN** it SHALL return `["joint_12", "joint_13", "joint_14", "joint_15"]`

#### Scenario: Index finger joints
- **WHEN** accessing `finger_joint_names["index"]`
- **THEN** it SHALL return `["joint_0", "joint_1", "joint_2", "joint_3"]`

#### Scenario: Middle finger joints
- **WHEN** accessing `finger_joint_names["middle"]`
- **THEN** it SHALL return `["joint_4", "joint_5", "joint_6", "joint_7"]`

#### Scenario: Ring finger joints
- **WHEN** accessing `finger_joint_names["ring"]`
- **THEN** it SHALL return `["joint_8", "joint_9", "joint_10", "joint_11"]`

### Requirement: Forward kinematics
The Allegro hand SHALL compute forward kinematics for fingertip positions.

#### Scenario: FK computation
- **WHEN** calling `forward_kinematics(config)`
- **THEN** it SHALL return a dict with fingertip link transforms

#### Scenario: Fingertip links
- **WHEN** computing FK
- **THEN** the result SHALL include transforms for all 4 fingertip links

### Requirement: Cost function implementation
The Allegro hand SHALL build appropriate costs for fingertip retargeting.

#### Scenario: Position matching cost
- **WHEN** building costs with target fingertip positions
- **THEN** it SHALL include position costs matching target to fingertip links

#### Scenario: Joint limit cost
- **WHEN** building costs
- **THEN** it SHALL include `limit_cost` to respect URDF joint limits

#### Scenario: Rest cost
- **WHEN** building costs with `q_current` provided
- **THEN** it SHALL include `rest_cost` for smooth motion
