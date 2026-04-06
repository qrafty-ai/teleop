# Tasks: Dexterous Hand Motion Retargeting

## 1. Python Interface Models

- [ ] 1.1 Create `teleop_xr/ik/hand_types.py` with Pydantic models: `HandJoint`, `HandPose`, `HandPosesMessage`
- [ ] 1.2 Add validation for 25 joints, coordinate ranges, and required fields
- [ ] 1.3 Add `get_fingertip_positions()` helper method to `HandPose`
- [ ] 1.4 Add unit tests for Pydantic models

## 2. BaseHandRobot Interface

- [ ] 2.1 Create `teleop_xr/ik/robot_hand.py` with `BaseHandRobot` abstract class
- [ ] 2.2 Implement `finger_joint_names` property returning `dict[str, list[str]]`
- [ ] 2.3 Implement `handedness` property returning `Literal["left", "right", "both"]`
- [ ] 2.4 Implement `hand_dof` property calculating total DOF from finger joints
- [ ] 2.5 Add abstract `build_hand_costs(fingertip_positions, q_current)` method
- [ ] 2.6 Add docstrings and type hints

## 3. Allegro Hand Robot

- [ ] 3.1 Create `teleop_xr/ik/robots/allegro.py` with `AllegroHandRobot` class
- [ ] 3.2 Implement `_load_default_urdf()` using RAM to fetch from dexsuite/dex-urdf
- [ ] 3.3 Define `finger_joint_names` with correct joint_0-15 mapping
- [ ] 3.4 Implement `forward_kinematics()` returning fingertip transforms
- [ ] 3.5 Implement `build_costs()` using standard cost stack
- [ ] 3.6 Implement `build_hand_costs()` with fingertip position matching
- [ ] 3.7 Add entry point to `pyproject.toml`: `"allegro-hand" = "teleop_xr.ik.robots.allegro:AllegroHandRobot"`
- [ ] 3.8 Test URDF loading and FK computation

## 4. Hand Retargeting Solver

- [ ] 4.1 Create `teleop_xr/ik/hand_solver.py` with `HandRetargetingSolver` class
- [ ] 4.2 Implement `__init__(hand_robot: BaseHandRobot)`
- [ ] 4.3 Implement `solve(fingertip_positions, q_current)` using jaxls LM optimizer
- [ ] 4.4 Add position matching costs for 5 fingertips
- [ ] 4.5 Add joint limit and rest costs
- [ ] 4.6 Return `(root_transform, joint_angles)` tuple
- [ ] 4.7 Add performance logging (solve time tracking)
- [ ] 4.8 Test solver with sample fingertip positions

## 5. WebSocket Protocol Extension

- [ ] 5.1 Update WebSocket message handler in `teleop_xr/__init__.py` to accept `hand_poses` type
- [ ] 5.2 Parse incoming messages using `HandPosesMessage` Pydantic model
- [ ] 5.3 Extract 5 fingertip positions from 25 joints
- [ ] 5.4 Apply RUB→FLU coordinate transformation
- [ ] 5.5 Call `HandRetargetingSolver.solve()` with converted positions
- [ ] 5.6 Log hand pose reception (similar to existing console logging)
- [ ] 5.7 Handle validation errors gracefully (skip frame, log error)

## 6. Viser Debug Visualization

- [ ] 6.1 Create `teleop_xr/demo/hand_visualizer.py` with `HandDebugVisualizer` class
- [ ] 6.2 Implement `__init__(port)` to start viser server
- [ ] 6.3 Implement `update_target_hand(joints)` to draw blue spheres at joint positions
- [ ] 6.4 Implement `update_robot_hand(joint_angles, fk_result)` to draw green robot hand
- [ ] 6.5 Implement `show_errors(target, achieved)` to draw red error lines
- [ ] 6.6 Add joint angle text overlay in viser GUI
- [ ] 6.7 Add scene setup (grid, lighting, camera position)

## 7. Hand Demo Mode

- [ ] 7.1 Add `--hand` argument to `teleop_xr/demo/__main__.py` CLI
- [ ] 7.2 Create `run_hand_demo()` function in `teleop_xr/demo/hand_demo.py`
- [ ] 7.3 Initialize `AllegroHandRobot` and `HandRetargetingSolver`
- [ ] 7.4 Initialize `HandDebugVisualizer` on port 8080
- [ ] 7.5 Start WebSocket server to receive hand tracking data
- [ ] 7.6 Main loop: receive → validate → solve → visualize
- [ ] 7.7 Print viser URL to console on startup
- [ ] 7.8 Add keyboard shortcuts (Q to quit, R to reset)
- [ ] 7.9 Add FPS/solve time display in TUI

## 8. Frontend Hand Tracking

- [ ] 8.1 Create `webxr/src/xr/hand_system.ts` with `HandSystem` class
- [ ] 8.2 Implement hand joint capture from WebXR (25 joints per hand)
- [ ] 8.3 Format joints with name, position, orientation, radius
- [ ] 8.4 Send `hand_poses` WebSocket messages with right hand data only
- [ ] 8.5 Add timestamp to each message
- [ ] 8.6 Handle hand tracking unavailable (graceful skip)
- [ ] 8.7 Register system in `webxr/src/xr/index.ts`

## 9. Testing & Validation

- [ ] 9.1 Test Allegro URDF loads correctly via RAM
- [ ] 9.2 Test forward kinematics produces expected fingertip positions
- [ ] 9.3 Test solver converges for sample hand poses
- [ ] 9.4 Test viser visualization displays correctly
- [ ] 9.5 Test WebSocket message parsing with Pydantic
- [ ] 9.6 Test coordinate transformation (RUB→FLU)
- [ ] 9.7 Run full demo with Quest headset
- [ ] 9.8 Verify solve time < 20ms per frame on GPU

## 10. Documentation

- [ ] 10.1 Update README with `--hand` mode instructions
- [ ] 10.2 Document Pydantic models in code
- [ ] 10.3 Add docstrings to all public methods
- [ ] 10.4 Update API documentation for new WebSocket message type
- [ ] 10.5 Add troubleshooting section for common issues
- [ ] 10.6 Update AGENTS.md with new module locations

## 11. Integration & Cleanup

- [ ] 11.1 Run linter and type checker on all new files
- [ ] 11.2 Ensure no `type: ignore` or `as any` suppressions
- [ ] 11.3 Verify all tests pass
- [ ] 11.4 Check for unused imports
- [ ] 11.5 Verify pyproject.toml entry points are correct
- [ ] 11.6 Update .gitignore if needed
- [ ] 11.7 Final code review
