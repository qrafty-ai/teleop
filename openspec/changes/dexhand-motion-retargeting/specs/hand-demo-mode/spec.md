# Spec: Hand Demo Mode

## ADDED Requirements

### Requirement: --hand CLI flag

The demo SHALL support a `--hand` mode for hand retargeting demonstration.

#### Scenario: Mode selection

- **WHEN** running `python -m teleop_xr.demo --hand`
- **THEN** it SHALL start the hand teleoperation demo

#### Scenario: Help text

- **WHEN** running with `--help`
- **THEN** `--hand` SHALL be documented as "Hand retargeting demo mode"

### Requirement: WebSocket server for hand demo

The demo SHALL start a WebSocket server to receive hand tracking data.

#### Scenario: Server startup

- **WHEN** starting the hand demo
- **THEN** it SHALL start a WebSocket server on the configured port

#### Scenario: Hand pose reception

- **WHEN** receiving `hand_poses` messages
- **THEN** it SHALL parse and validate them using Pydantic models

### Requirement: Viser debug visualization

The demo SHALL launch a viser server for 3D visualization of hand retargeting.

#### Scenario: Viser startup

- **WHEN** starting the hand demo
- **THEN** it SHALL launch viser on port 8080 (or configurable)

#### Scenario: Visualization URL

- **WHEN** viser starts
- **THEN** the demo SHALL print the viser URL (e.g., `http://localhost:8080`)

#### Scenario: Target hand visualization

- **WHEN** receiving hand tracking data
- **THEN** viser SHALL display incoming hand joints as blue spheres

#### Scenario: Robot hand visualization

- **WHEN** solving retargeting
- **THEN** viser SHALL display the solved Allegro hand pose in green

### Requirement: Fingertip error visualization

The demo SHALL visualize retargeting errors as lines.

#### Scenario: Error display

- **WHEN** comparing target to achieved fingertip positions
- **THEN** viser SHALL draw red lines from target to achieved positions

#### Scenario: Real-time update

- **WHEN** new hand data arrives
- **THEN** the error visualization SHALL update in real-time

### Requirement: Joint angle display

The demo SHALL display current joint angles in the TUI or viser GUI.

#### Scenario: Joint values

- **WHEN** solving produces joint angles
- **THEN** they SHALL be displayed as text overlay in viser

#### Scenario: Angle format

- **WHEN** displaying angles
- **THEN** they SHALL be shown in degrees with 2 decimal places

### Requirement: Pydantic validation

The demo SHALL use Pydantic models to validate incoming hand pose messages.

#### Scenario: Valid message

- **WHEN** receiving a well-formed `hand_poses` message
- **THEN** it SHALL parse into `HandPosesMessage` model successfully

#### Scenario: Invalid message

- **WHEN** receiving a malformed message
- **THEN** it SHALL log the error and skip the frame

#### Scenario: Joint count validation

- **WHEN** a message has fewer than 25 joints
- **THEN** validation SHALL fail with clear error message

### Requirement: Coordinate conversion

The demo SHALL convert coordinates from WebXR RUB to ROS2 FLU.

#### Scenario: Position conversion

- **WHEN** processing hand joint positions
- **THEN** it SHALL apply the standard RUB→FLU transform before solving

#### Scenario: Consistency with existing code

- **WHEN** converting coordinates
- **THEN** it SHALL use the same transform as `__convert_pose_to_ros`
