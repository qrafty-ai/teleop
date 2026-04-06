# Spec: WebXR Hand Tracking

## ADDED Requirements

### Requirement: Capture hand joints from WebXR

The frontend SHALL capture all 25 hand joints from WebXR hand tracking API when available.

#### Scenario: Hand tracking available

- **WHEN** the VR session supports hand tracking and user hands are visible
- **THEN** the system SHALL capture position, orientation, and radius for all 25 joints per hand

#### Scenario: Hand tracking unavailable

- **WHEN** hand tracking is not available or hands are not visible
- **THEN** the system SHALL gracefully degrade and not send hand pose messages

### Requirement: Hand joint data format

The frontend SHALL format hand joint data with joint name, position (x,y,z), orientation (quaternion), and radius.

#### Scenario: Joint data structure

- **WHEN** formatting a hand joint for transmission
- **THEN** the joint SHALL contain: `jointName` (string), `position` ({x, y, z}), `orientation` ({x, y, z, w}), `radius` (float)

#### Scenario: Coordinate system

- **WHEN** capturing hand positions
- **THEN** coordinates SHALL be in WebXR reference space (meters, RUB convention)

### Requirement: Right-hand only in v1

The frontend SHALL capture and transmit only the right hand in v1.

#### Scenario: Right hand detected

- **WHEN** the user's right hand is visible
- **THEN** the system SHALL send `hand_poses` message with `handedness: "right"`

#### Scenario: Left hand only

- **WHEN** only the left hand is visible
- **THEN** the system SHALL NOT send hand pose messages

### Requirement: WebSocket transmission

The frontend SHALL transmit hand poses via WebSocket as `hand_poses` messages.

#### Scenario: Message format

- **WHEN** sending hand data to backend
- **THEN** the message SHALL have `type: "hand_poses"` and `data` containing `handedness`, `timestamp`, and `joints` array

#### Scenario: Transmission rate

- **WHEN` streaming hand data
- **THEN** messages SHALL be sent at the XR frame rate (typically 60-90 Hz)

### Requirement: Timestamp inclusion

Each hand pose message SHALL include a millisecond timestamp.

#### Scenario: Timestamp format

- **WHEN** creating a hand pose message
- **THEN** the `timestamp` field SHALL be an integer representing milliseconds since Unix epoch
