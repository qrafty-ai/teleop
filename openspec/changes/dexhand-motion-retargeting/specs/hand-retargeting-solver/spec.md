# Spec: Hand Retargeting Solver

## ADDED Requirements

### Requirement: HandRetargetingSolver class

The system SHALL provide a `HandRetargetingSolver` class for hand motion retargeting.

#### Scenario: Solver initialization

- **WHEN** creating a solver
- **THEN** it SHALL accept a `BaseHandRobot` instance

#### Scenario: Import path

- **WHEN** importing the solver
- **THEN** `HandRetargetingSolver` SHALL be available from `teleop_xr.ik.hand_solver`

### Requirement: Solve method signature

The solver SHALL expose a `solve` method accepting fingertip positions and returning joint angles.

#### Scenario: Input format

- **WHEN** calling `solve`
- **THEN** the first argument SHALL be `fingertip_positions: jax.Array` with shape `(num_fingers, 3)`

#### Scenario: Optional current config

- **WHEN** calling `solve`
- **THEN** it SHALL accept optional `q_current: jax.Array | None` for smoothness regularization

#### Scenario: Return values

- **WHEN** solving succeeds
- **THEN** it SHALL return `tuple[jaxlie.SE3, jax.Array]` (root_transform, joint_angles)

### Requirement: Levenberg-Marquardt optimization

The solver SHALL use Levenberg-Marquardt optimization via pyroki/jaxls.

#### Scenario: Optimizer selection

- **WHEN** solving the retargeting problem
- **THEN** it SHALL use `jaxls.LeastSquaresProblem` with LM optimizer

#### Scenario: Cost composition

- **WHEN** building the optimization problem
- **THEN** it SHALL use costs from `hand_robot.build_hand_costs()`

### Requirement: Fingertip position matching

The solver SHALL match 5 fingertip positions from input to robot fingertips.

#### Scenario: Thumb matching

- **WHEN** solving with thumb position
- **THEN** the solution SHALL minimize distance to Allegro thumb fingertip

#### Scenario: All fingers matching

- **WHEN** solving with 5 fingertip positions
- **THEN** the solution SHALL minimize sum of squared distances to all 4 Allegro fingertips

#### Scenario: Pinky handling

- **WHEN** input includes pinky position
- **THEN** it SHALL be matched to Allegro's 4th (ring) finger positionally

### Requirement: Single-frame solving

The solver SHALL solve each frame independently for real-time performance.

#### Scenario: Frame independence

- **WHEN** solving consecutive frames
- **THEN** each frame SHALL be solved without reference to previous solutions

#### Scenario: Performance requirement

- **WHEN** running on GPU
- **THEN** solve time SHALL be under 20ms per frame

### Requirement: Coordinate transformation

The solver SHALL handle coordinate system conversion.

#### Scenario: Input coordinates

- **WHEN** receiving fingertip positions
- **THEN** it SHALL assume input is in ROS2 FLU coordinates (converted from WebXR RUB by backend)

#### Scenario: Output coordinates

- **WHEN** returning joint angles
- **THEN** they SHALL be in the robot's native configuration space
