# Design: Dexterous Hand Motion Retargeting

## Context

### Current State
teleop_xr currently supports arm/whole-body IK through a well-established pattern:
- Robot models inherit from `BaseRobot` in `teleop_xr/ik/robot.py`
- Robots register via `pyproject.toml` entry points under `teleop_xr.robots`
- URDFs load via the Resource Asset Manager (RAM) from GitHub repos
- WebXR streams controller/head poses via WebSocket to Python backend
- Pyroki solves IK using Levenberg-Marquardt optimization

### What's Missing
There is no hand tracking support. Users cannot control robot hands dexterously—they only have gripper open/close via controller triggers.

### Constraints
- Must follow existing `BaseRobot` patterns
- Must use RAM for URDF loading (no local copies)
- Must integrate with existing WebSocket protocol
- Must work with pyroki's cost-based optimization framework
- WebXR hand tracking is already enabled in the frontend (`handTracking: true`)

---

## Goals / Non-Goals

**Goals:**
1. Capture WebXR hand tracking data (25 joints) and stream to Python backend
2. Create `BaseHandRobot` interface extending `BaseRobot` for hand-specific functionality
3. Implement Allegro Hand (16 DOF) as the first supported hand robot
4. Adapt pyroki's keypoint-based retargeting for real-time hand control
5. Add `--hand` demo mode for interactive hand teleoperation
6. Design for future bimanual and arm+hand composition

**Non-Goals:**
- Shadow Hand or Robotiq support (future work)
- Left-hand support in v1 (design for it, but implement right-hand only)
- Contact detection or force feedback
- Sliding window / batch processing (single-frame only)
- Simultaneous arm IK + hand solving (design for composition, but not implement yet)

---

## Decisions

### Decision 1: Right-Hand-Only Initial Implementation

**Choice:** Implement right-hand retargeting first. Design interfaces to allow left-hand addition without breaking changes.

**Rationale:**
- Reduces complexity for v1
- Most users are right-handed
- Can validate approach before doubling implementation

**Alternative considered:** Bimanual from start — rejected due to complexity and unknowns in finger mapping.

**Interface Design:**
```python
class BaseHandRobot(BaseRobot):
    @property
    @abstractmethod
    def handedness(self) -> Literal["left", "right", "both"]:
        """Which hand(s) this robot model supports."""
```

---

### Decision 2: Fingertip-Only Keypoint Mapping

**Choice:** Map only 5 fingertip positions (thumb, index, middle, ring, pinky) from WebXR to robot, not all 25 joints.

**Rationale:**
- WebXR provides 25 joints, but Allegro Hand has 16 DOF with different kinematic structure
- Direct joint-to-joint mapping is morphologically impossible (different finger counts, joint axes)
- Pyroki's algorithm works with arbitrary keypoint sets—we provide target fingertip positions, it solves for joint angles
- Simpler to implement and reason about
- Can add intermediate joint targets later if needed

**Alternative considered:** Full 25-joint mapping — rejected because Allegro's 4 fingers × 4 joints doesn't map cleanly to human 5 fingers × 4 joints.

**Mapping Strategy:**
| WebXR Joint | Robot Target |
|-------------|--------------|
| `thumb-tip` | Thumb fingertip position |
| `index-finger-tip` | Index fingertip position |
| `middle-finger-tip` | Middle fingertip position |
| `ring-finger-tip` | Ring fingertip position |
| `pinky-finger-tip` | Pinky fingertip position |

The pinky will be ignored or mapped to Allegro's 4th finger positionally.

---

### Decision 3: Single-Frame Real-Time Solving

**Choice:** Solve each frame independently (~15ms) without temporal smoothing.

**Rationale:**
- Pyroki's LM solver is fast enough (~5-15ms/frame on GPU)
- Teleoperation benefits from low latency over smoothness
- Users can visually smooth motion themselves
- Simpler implementation (no buffer management)

**Alternative considered:** Sliding window optimization — rejected due to added latency (N frames of delay) and complexity.

---

### Decision 4: Extend BaseRobot, Don't Replace

**Choice:** Create `BaseHandRobot` that inherits from `BaseRobot` and adds hand-specific methods.

**Rationale:**
- Hands are still robots (have URDF, joints, FK, costs)
- Can reuse existing infrastructure (RAM loading, entry points, cost functions)
- Future `ComposedRobot` can hold both `BaseRobot` (arm) and `BaseHandRobot` (hand)

**Interface:**
```python
class BaseHandRobot(BaseRobot):
    """Base class for dexterous hands and grippers."""

    @property
    @abstractmethod
    def finger_joint_names(self) -> dict[str, list[str]]:
        """Map finger names to joint names.
        Example: {'thumb': ['joint_0', 'joint_1', 'joint_2', 'joint_3'], ...}
        """

    @abstractmethod
    def build_hand_costs(
        self,
        fingertip_positions: jax.Array,  # (num_fingers, 3)
        q_current: jax.Array | None
    ) -> list[Cost]:
        """Build costs for fingertip position matching."""

    @property
    def hand_dof(self) -> int:
        """Total DOF for hand."""
        return sum(len(joints) for joints in self.finger_joint_names.values())
```

---

### Decision 5: Allegro Hand via dexsuite/dex-urdf

**Choice:** Use Allegro Hand from dexsuite/dex-urdf repository (MIT license).

**Rationale:**
- MIT license (compatible with project)
- GLB meshes (modern, small)
- 16 DOF is a good balance (4 fingers × 4 joints)
- ROS/MoveIt support (well-tested)
- Available via RAM

**URDF Loading:**
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

**Finger Mapping for Allegro:**
```python
finger_joint_names = {
    'thumb': ['joint_12', 'joint_13', 'joint_14', 'joint_15'],
    'index': ['joint_0', 'joint_1', 'joint_2', 'joint_3'],
    'middle': ['joint_4', 'joint_5', 'joint_6', 'joint_7'],
    'ring': ['joint_8', 'joint_9', 'joint_10', 'joint_11'],
}
```

---

### Decision 6: Separate Hand Retargeting Solver

**Choice:** Create `HandRetargetingSolver` class rather than adding to existing `IKSolver`.

**Rationale:**
- Different problem: IK matches SE(3) poses; retargeting matches keypoint positions
- Different cost functions: retargeting uses position-only costs, not full pose costs
- Can be composed later with IK solver
- Easier to test and debug independently

**Solver Interface:**
```python
class HandRetargetingSolver:
    def __init__(self, hand_robot: BaseHandRobot):
        self.hand_robot = hand_robot

    def solve(
        self,
        fingertip_positions: jax.Array,  # (num_fingers, 3) in world frame
        q_current: jax.Array | None = None,
    ) -> tuple[jaxlie.SE3, jax.Array]:
        """Returns: (root_transform, joint_angles)"""
```

---

### Decision 7: WebSocket Protocol Extension

**Choice:** Add `hand_poses` message type alongside existing `xr_state`.

**Rationale:**
- Hand data is conceptually separate from controller data
- Different update frequency possible
- Easier to filter/process on backend

**Message Format:**
```typescript
// Frontend → Backend
{
  type: "hand_poses",
  data: {
    handedness: "right",
    timestamp: 1234567890,
    joints: [
      {
        jointName: "thumb-tip",
        position: { x: 0.1, y: 0.2, z: -0.3 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
        radius: 0.01
      },
      // ... 24 more joints
    ]
  }
}
```

**Backend Processing:**
- Extract 5 fingertip positions
- Convert from WebXR RUB to ROS2 FLU coordinates
- Pass to `HandRetargetingSolver`
- Return joint angles to robot controller

---

### Decision 8: `--hand` Demo Mode Structure

**Choice:** Mirror existing `--mode ik` pattern with hand-specific visualization.

**Rationale:**
- Consistent with existing demo modes
- Can reuse TUI infrastructure
- Clear separation from arm IK demo

**Demo Features:**
- Visualize incoming hand joint positions
- Display solved joint angles
- Show fingertip position errors (target vs achieved)
- Toggle between teleoperation and scripted demos

---

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Finger mapping may not feel natural** | Medium | Start with 5 fingertips; can add intermediate joints later if users report unnatural poses |
| **Allegro Hand URDF may have issues** | Low | dexsuite/dex-urdf is actively maintained; fallback to testing with viser visualization |
| **Performance may degrade on CPU** | Medium | Document GPU recommendation; provide quality settings (reduce iterations) |
| **WebXR hand tracking availability** | Low | Check `handTracking` feature before enabling; graceful fallback to controller mode |
| **Coordinate system confusion** | Medium | Document RUB→FLU transform clearly; add debug visualization |
| **No tactile feedback** | High | Document limitation; future work could add predicted contact visualization |

---

## Migration Plan

This is additive functionality—no migration needed for existing users.

**Deployment Steps:**
1. Merge hand tracking frontend changes
2. Deploy new Python package with Allegro Hand support
3. Update documentation with `--hand` mode instructions
4. Announce in release notes

**Rollback:**
- Frontend: Feature flag `handTracking` already exists (set to `true` currently)
- Backend: New code paths only activated when hand messages received

---

## Open Questions

1. **Finger Mapping Detail:** Need to verify exact Allegro joint names from URDF and confirm pinky→4th finger mapping strategy.

2. **Root Transform:** Should the hand base transform be fixed (mounted on arm) or solved (free-floating)? For v1, assume fixed base; solve only finger joints.

3. **Visualization:** Should we add a 3D hand visualization in the demo TUI (using viser)?

4. **Joint Limits:** Does Allegro URDF include proper joint limits? May need manual tuning.


---

### Decision 9: Pydantic Interface for Hand Data

**Choice:** Use Pydantic models to define and validate hand joint transmission interface.

**Rationale:**
- Type safety at runtime for WebSocket messages
- Automatic validation of hand pose data structure
- Self-documenting code (schema is code)
- Easy serialization/deserialization
- Consistent with modern Python async patterns

**Pydantic Models:**

```python
from pydantic import BaseModel, Field
from typing import Literal

class HandJoint(BaseModel):
    """Single hand joint from WebXR."""
    joint_name: str = Field(..., description="WebXR joint name")
    position: tuple[float, float, float] = Field(..., description="Position in meters (x, y, z)")
    orientation: tuple[float, float, float, float] = Field(..., description="Quaternion (x, y, z, w)")
    radius: float = Field(..., description="Joint radius in meters")

class HandPose(BaseModel):
    """Complete hand pose with all 25 joints."""
    handedness: Literal["left", "right"] = Field(..., description="Which hand")
    timestamp: int = Field(..., description="Timestamp in milliseconds")
    joints: list[HandJoint] = Field(..., min_length=25, max_length=25, description="All 25 hand joints")

    def get_fingertip_positions(self) -> dict[str, tuple[float, float, float]]:
        """Extract 5 fingertip positions for retargeting."""
        tip_names = [
            "thumb-tip",
            "index-finger-tip",
            "middle-finger-tip",
            "ring-finger-tip",
            "pinky-finger-tip"
        ]
        return {
            joint.joint_name: joint.position
            for joint in self.joints
            if joint.joint_name in tip_names
        }

class HandPosesMessage(BaseModel):
    """WebSocket message wrapper."""
    type: Literal["hand_poses"] = "hand_poses"
    data: HandPose
```

**Validation Benefits:**
- Ensures exactly 25 joints received
- Validates coordinate ranges
- Type-safe access to position/orientation data
- Clear error messages on malformed input

---

### Decision 10: Viser Debug Visualization

**Choice:** Launch viser 3D visualization server in `--hand` demo mode for debugging hand poses.

**Rationale:**
- Viser is already a project dependency (used in viser mode)
- Real-time 3D visualization is essential for debugging retargeting
- Can visualize both incoming hand joints and solved robot hand
- Side-by-side comparison of target vs achieved fingertip positions

**Visualization Features:**

```python
class HandDebugVisualizer:
    """Debug visualization for hand retargeting using viser."""

    def __init__(self, port: int = 8080):
        self.server = viser.ViserServer(port=port)
        self._setup_scene()

    def _setup_scene(self):
        """Initialize visualization elements."""
        # Target hand (from WebXR) - shown in blue
        self.target_hand = self.server.scene.add_group("target_hand")

        # Solved robot hand (Allegro) - shown in green
        self.robot_hand = self.server.scene.add_group("robot_hand")

        # Fingertip position errors - shown as red lines
        self.error_lines = self.server.scene.add_group("errors")

        # Joint angle display (text overlay)
        self.joint_angles_gui = self.server.gui.add_text("Joint Angles", initial_value="")

    def update_target_hand(self, joints: list[HandJoint]):
        """Update visualization of incoming hand tracking data."""
        for joint in joints:
            # Draw spheres at joint positions
            self.server.scene.add_sphere(
                name=f"target_{joint.joint_name}",
                position=joint.position,
                radius=joint.radius,
                color=(0.3, 0.5, 1.0),  # Blue
                parent=self.target_hand
            )

    def update_robot_hand(self, joint_angles: jax.Array, fk_result: dict):
        """Update visualization of solved robot hand pose."""
        # Draw robot links and joints
        # Show fingertip positions from FK
        pass

    def show_errors(self, target_positions: dict, achieved_positions: dict):
        """Draw lines showing retargeting errors."""
        for finger_name in target_positions:
            target = target_positions[finger_name]
            achieved = achieved_positions[finger_name]
            # Draw red line from target to achieved
            self.server.scene.add_line(
                name=f"error_{finger_name}",
                start=target,
                end=achieved,
                color=(1.0, 0.0, 0.0),  # Red
                line_width=2.0
            )
```

**Demo Mode Integration:**

```python
# In teleop_xr/demo/__main__.py --hand mode
def run_hand_demo():
    visualizer = HandDebugVisualizer(port=8080)
    print(f"Viser debug visualization: http://localhost:8080")

    # Main loop
    while running:
        hand_pose = receive_hand_pose_from_websocket()

        # Update visualization with incoming data
        visualizer.update_target_hand(hand_pose.joints)

        # Solve retargeting
        root_transform, joint_angles = solver.solve(
            hand_pose.get_fingertip_positions()
        )

        # Update visualization with solution
        visualizer.update_robot_hand(joint_angles, fk_result)

        # Show errors
        visualizer.show_errors(
            target=hand_pose.get_fingertip_positions(),
            achieved=compute_fk_fingertips(joint_angles)
        )
```

**Benefits:**
- Immediate visual feedback on retargeting quality
- Debug coordinate system issues
- Validate finger mapping visually
- Compare target vs achieved poses in real-time

---

## Open Questions - Status

| Question | Status | Decision / Notes |
|----------|--------|------------------|
| **1. Finger Mapping Strategy** | ✅ **DECIDED** | Use 5 fingertip positions only (thumb, index, middle, ring, pinky). Pinky maps to Allegro's 4th finger positionally. |
| **2. Bimanual Support** | ✅ **DECIDED** | Right-hand only v1. Interface supports left-hand extension. |
| **3. Contact Detection** | ✅ **DECIDED** | No contact detection in v1. Pure kinematic retargeting. |
| **4. Real-time vs Batch** | ✅ **DECIDED** | Single-frame real-time solving (~15ms). |
| **5. Visualization** | ✅ **DECIDED** | Use viser for debug visualization in `--hand` demo mode. Shows target hand, solved robot hand, and error lines. |
| **6. Python Interface** | ✅ **DECIDED** | Use Pydantic models (`HandJoint`, `HandPose`, `HandPosesMessage`) for type-safe WebSocket message handling. |
| **7. Root Transform** | 🔄 **PENDING** | Fixed base for v1 (mounted hand). Free-floating can be added later if needed. |
| **8. Joint Limits** | 🔄 **PENDING** | Verify Allegro URDF limits; may need manual tuning. |
