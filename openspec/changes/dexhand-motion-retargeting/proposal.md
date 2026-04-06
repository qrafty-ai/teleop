# Proposal: Dexterous Hand Motion Retargeting

## Why

Currently, teleop_xr only supports arm/whole-body IK but lacks dexterous hand control. Users in VR cannot control robot hands with the same fidelity as their own hands. This limits teleoperation tasks requiring fine manipulation (grasping, tool use, contact-rich operations). By adding hand motion retargeting using WebXR hand tracking and pyroki's proven algorithms, we enable high-DOF hand control that matches the project's existing architecture.

## What Changes

- **New WebXR hand tracking system** — Capture 25 joints per hand from Quest/VR headsets and stream to Python backend
- **New `BaseHandRobot` interface** — Unified abstraction supporting dexterous hands (16-20 DOF) and parallel grippers (1 DOF)
- **New hand robot implementations** — Allegro Hand (16 DOF) as primary, Shadow Hand (20 DOF) as secondary, Robotiq 2F-85 (1 DOF) as gripper fallback
- **Hand retargeting solver** — Adapt pyroki's Levenberg-Marquardt keypoint-based retargeting for real-time use
- **New `--hand` demo mode** — Interactive TUI demonstration of hand retargeting
- **WebSocket protocol extension** — Add hand pose messages alongside existing body pose streaming
- **Unified robot composition** — Design for future integration where arm IK and hand retargeting run simultaneously

## Capabilities

### New Capabilities
- `webxr-hand-tracking`: WebXR hand tracking data capture and streaming
- `hand-robot-interface`: Base hand robot abstraction and joint mapping
- `allegro-hand`: Allegro Hand (16 DOF) robot model and URDF loading via RAM
- `hand-retargeting-solver`: Pyroki-based hand motion retargeting algorithm
- `hand-demo-mode`: Interactive CLI demo for hand teleoperation

### Modified Capabilities
- *(none — this is additive functionality)*

## Impact

**Affected Code:**
- `teleop_xr/ik/` — New hand robot module, retargeting solver
- `teleop_xr/demo/` — New `--hand` mode implementation
- `webxr/src/xr/` — Hand tracking system, WebSocket message types
- `pyproject.toml` — New robot entry points

**Dependencies:**
- Uses existing `pyroki` for retargeting (already in project)
- Uses existing RAM module for URDF fetching
- Uses existing WebSocket infrastructure

**API Changes:**
- New WebSocket message type: `hand_poses` with joint positions per finger
- New robot entry point pattern following existing `teleop_xr.robots` convention

---

## Research Summary

### Robot Patterns (teleop_xr/ik/robots/)

**Registration:** Via `pyproject.toml` entry points under `[project.entry-points."teleop_xr.robots"]`

**BaseRobot Interface:**
- Abstract methods: `_load_default_urdf()`, `forward_kinematics()`, `build_costs()`, `get_default_config()`
- Properties: `actuated_joint_names`, `joint_var_cls`, `orientation`, `supported_frames`
- URDF loading via RAM: `ram.get_resource(repo_url, path_inside_repo, xacro_args, resolve_packages)`

**Cost Functions:**
- `rest_cost` — Regularization toward current/rest pose
- `manipulability_cost` — Avoid singularities
- `pose_cost_analytic_jac` — End-effector tracking
- `limit_cost` — Joint bounds
- `self_collision_cost` — Collision avoidance (optional)

### Dexterous Hand URDF Sources

| Hand | Repository | License | DOF | Notes |
|------|------------|---------|-----|-------|
| **Allegro Hand** (recommended) | [dexsuite/dex-urdf](https://github.com/dexsuite/dex-urdf) | MIT | 16 (4×4) | GLB meshes, ROS support |
| Shadow Hand | [shadow-robot/sr_common](https://github.com/shadow-robot/sr_common) | GPL-3.0 | 20 (5×4) | Most complete, license concern |
| Robotiq 2F-85 | [a-price/robotiq_arg85_description](https://github.com/a-price/robotiq_arg85_description) | Unknown | 1 | Parallel gripper fallback |

### Pyroki Hand Retargeting Algorithm

**Technique:** Levenberg-Marquardt nonlinear least squares via `jaxls`

**Input:** Hand joint keypoints (3D positions), optional contact information
**Output:** Robot joint angles, root transform (SE3)

**Key Cost Functions:**
1. **Local Alignment** (weight 10.0) — Match relative joint positions/angles
2. **Global Alignment** (weight 1.0) — Match absolute keypoint positions
3. **Contact** (weight 5.0) — Preserve fingertip-object contact
4. **Smoothness** (weight 2.0) — Penalize rapid joint changes

**Performance:** ~5-15ms per frame on GPU (real-time capable)

**Mathematical Approach:**
- Vector retargeting: `delta_mano = mano_pos[:, None] - mano_pos[None, :]`
- Cosine similarity for angles: `1 - (delta_mano_norm * delta_robot_norm).sum()`
- Learnable scale factors for morphology differences

**Integration Notes:**
- Replace DexYCB/MANO input with WebXR hand tracking data
- Create finger-to-joint mapping for target robot
- Coordinate transform: WebXR (RUB) → ROS2/Pyroki (FLU)

### WebXR Hand Tracking

**API:** `navigator.xr.requestSession('immersive-ar', { optionalFeatures: ['hand-tracking'] })`

**Data Format:**
- 25 joints per hand (wrist + 4 joints × 5 fingers + 4 finger tips)
- `XRJointPose` with `transform.position` and `transform.orientation`
- Coordinate space: WebXR reference space (RUB convention)

**Joint Hierarchy:**
```
wrist
├── thumb-metacarpal → thumb-phalanx-proximal → thumb-phalanx-distal → thumb-tip
├── index-finger-metacarpal → index-finger-phalanx-proximal → index-finger-phalanx-intermediate → index-finger-phalanx-distal → index-finger-tip
├── middle-finger-* (same structure)
├── ring-finger-* (same structure)
└── pinky-finger-* (same structure)
```

**Access Pattern:**
```typescript
const session = await navigator.xr.requestSession('immersive-ar', {
  optionalFeatures: ['hand-tracking']
});
// Per frame:
const frame = await session.requestAnimationFrame((time, frame) => {
  const referenceSpace = renderer.xr.getReferenceSpace();
  for (const inputSource of session.inputSources) {
    if (inputSource.hand) {
      for (const joint of inputSource.hand.values()) {
        const pose = frame.getJointPose(joint, referenceSpace);
        // pose.transform.position, pose.transform.orientation
      }
    }
  }
});
```

---

## Design Considerations

### Interface Design (Core Problem #1)

**Unified Hand Interface:**
```python
class BaseHandRobot(BaseRobot):
    """Abstract base for dexterous hands and grippers."""

    @property
    @abstractmethod
    def finger_joint_names(self) -> dict[str, list[str]]:
        """Map finger names to their joint names.
        Example: {'thumb': ['joint_0', 'joint_1', ...], ...}
        """

    @abstractmethod
    def build_hand_costs(self,
                        target_keypoints: jax.Array,  # (num_fingers, num_joints_per_finger, 3)
                        finger_contacts: dict[str, jax.Array] | None,
                        q_current: jax.Array | None) -> list[Cost]:
        """Build costs for hand retargeting."""

    @property
    def hand_dof(self) -> int:
        """Total degrees of freedom for hand."""
```

**Parallel Gripper Compatibility:**
- Single DOF represented as `{'gripper': ['finger_joint']}`
- Build costs use simplified position-matching (gripper width ↔ hand aperture)

### Robot Composition (Core Problem #3)

**Future Integration Design:**
```python
class ComposedRobot(BaseRobot):
    """Combines arm and hand robots for simultaneous solving."""

    def __init__(self, arm_robot: BaseRobot, hand_robot: BaseHandRobot):
        self.arm = arm_robot
        self.hand = hand_robot

    def build_costs(self, target_L, target_R, target_Head,
                    hand_keypoints, q_current):
        # Combine arm IK costs + hand retargeting costs
        costs = []
        costs.extend(self.arm.build_costs(target_L, target_R, target_Head, q_current))
        costs.extend(self.hand.build_hand_costs(hand_keypoints, None, q_current))
        return costs
```

### URDF Loading via RAM (Core Problem #2)

**Allegro Hand via dexsuite/dex-urdf:**
```python
def _load_default_urdf(self) -> yourdfpy.URDF:
    self.urdf_path = str(ram.get_resource(
        repo_url="https://github.com/dexsuite/dex-urdf.git",
        path_inside_repo="robots/hands/allegro_hand/allegro_hand_right_glb.urdf",
        resolve_packages=True,
    ))
    self.mesh_path = str(ram.get_repo("https://github.com/dexsuite/dex-urdf.git"))
    return yourdfpy.URDF.load(self.urdf_path)
```

---

## Open Questions

1. **Finger Mapping Strategy:** Should we map WebXR's 25 joints directly to robot joints, or use fingertip positions only (simpler but less expressive)?
2. **Bimanual Support:** Do we need simultaneous left+right hand retargeting, or single-hand mode first?
3. **Contact Detection:** Should the demo include simple contact detection (gripper force feedback visualization), or pure kinematic retargeting?
4. **Real-time vs Batch:** Single-frame solving (~15ms) vs sliding window smoothing (better quality, higher latency)?


---

## Open Questions - Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| **1. Finger Mapping Strategy** | **TBD - Need more info** | Pyroki example uses MANO-to-Shadow mapping with helper functions (`get_mapping_from_mano_to_shadow`). Need to investigate what specific joint correspondences pyroki expects. |
| **2. Bimanual Support** | **Single-hand first, extend later** | Implement right-hand retargeting as primary. Design interface to allow left-hand addition without breaking changes. |
| **3. Contact Detection** | **No contact detection in v1** | Pure kinematic retargeting only. Contact is complex (requires object geometry, force sensing). Can add later if needed. |
| **4. Real-time vs Batch** | **Real-time (single-frame)** | Single-frame solving (~15ms on GPU) meets real-time requirements. Sliding window adds latency and complexity not needed for initial implementation. |
