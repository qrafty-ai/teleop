# pyright: reportCallIssue=false
"""
Dexmate Vega humanoid models (URDF fetched with RAM).

Source: https://github.com/dexmate-ai/dexmate-urdf

Design note: Vega uses RAM loading
----------------------------------
Vega URDFs are resolved from the dexmate-urdf repository through
``teleop_xr.ram.get_resource``. This keeps robot onboarding aligned with other
RAM-backed robots in this project and avoids requiring an additional Python
package dependency in the runtime environment.

Joint policy (arm teleop defaults)
----------------------------------
- Freeze wheels when present: ``B_wheel_j1/j2``, ``R_wheel_j1/j2``,
  ``L_wheel_j1/j2``.
- Freeze dexterous finger chains by prefix (both hands):
  ``L_th_``, ``L_ff_``, ``L_mf_``, ``L_rf_``, ``L_lf_``, and their ``R_*``
  counterparts.
- Keep all other URDF joints actuated.

After freezing, mimic links targeting fixed joints are cleared to keep Pyroki's
actuated-joint assumptions valid.
"""

from __future__ import annotations

import sys
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy

from teleop_xr.ik.robot import BaseRobot, Cost
from teleop_xr import ram

_DEXMATE_URDF_REPO_URL = "https://github.com/dexmate-ai/dexmate-urdf.git"

_WHEEL_JOINTS = (
    "B_wheel_j1",
    "B_wheel_j2",
    "R_wheel_j1",
    "R_wheel_j2",
    "L_wheel_j1",
    "L_wheel_j2",
)

_HAND_JOINT_PREFIXES = (
    "L_th_",
    "L_ff_",
    "L_mf_",
    "L_rf_",
    "L_lf_",
    "R_th_",
    "R_ff_",
    "R_mf_",
    "R_rf_",
    "R_lf_",
)

_VEGA_STANDING_POSE: dict[str, float] = {
    "torso_j1": 0.0,
    "torso_j2": 0.0,
    "torso_j3": 0.0,
    "head_j1": 0.0,
    "head_j2": 0.0,
    "head_j3": 0.0,
    "L_arm_j1": -1.57079,
    "L_arm_j2": 0.0,
    "L_arm_j3": 0.0,
    "L_arm_j4": 0.0,
    "L_arm_j5": 0.0,
    "L_arm_j6": 0.0,
    "L_arm_j7": 0.0,
    "R_arm_j1": 1.57079,
    "R_arm_j2": 0.0,
    "R_arm_j3": 0.0,
    "R_arm_j4": 0.0,
    "R_arm_j5": 0.0,
    "R_arm_j6": 0.0,
    "R_arm_j7": 0.0,
}


def _vega_rest_energy_weight(joint_name: str) -> float:
    if joint_name.startswith("torso_j"):
        return 20.0
    if joint_name.startswith("head_j"):
        return 8.0
    return 5.0


def _vega_centering_weight(joint_name: str) -> float:
    if joint_name.startswith("torso_j"):
        return 0.5
    if joint_name.startswith("head_j"):
        return 0.1
    return 0.0


def _strip_mimics_to_fixed_joints(urdf: yourdfpy.URDF) -> None:
    """Mimic chains break if the driven joint is fixed; pyroki requires mimicked joints stay actuated."""
    fixed = {name for name, joint in urdf.joint_map.items() if joint.type == "fixed"}
    for joint in urdf.joint_map.values():
        if joint.mimic is not None and joint.mimic.joint in fixed:
            joint.mimic = None


def _dexmate_urdf_repo_path(variant: str) -> str:
    parts = variant.split(".")
    if len(parts) < 2:
        raise ValueError(
            "Dexmate variant must be a dotted humanoid path, "
            f"e.g. 'vega_1.vega_1_f5d6', got {variant!r}."
        )
    subdir = "/".join(parts[:-1])
    return f"robots/humanoid/{subdir}/{parts[-1]}.urdf"


class DexmateVegaRobot(BaseRobot):
    """
    IK model for Dexmate Vega humanoids shipped in ``dexmate-urdf``.

    End-effector links default to ``L_ee`` / ``R_ee``; head tracking uses ``head_l3``
    (same naming across vega_1 / vega_1u / vega_1p variants in current packages).

    Constructor kwargs (besides ``urdf_string``):

    - ``variant``: dotted import path under ``robots.humanoid``, e.g.
      ``vega_1.vega_1_f5d6``, ``vega_1u.vega_1u_f5d6``, ``vega_1p.vega_1p_gripper``.
    - ``freeze_wheels``: fix wheel joints when present (full-body Vega).
    - ``freeze_hands``: fix dexterous finger joints (recommended for arm-teleop IK).
    - ``left_ee_link``, ``right_ee_link``, ``head_link``: override link names if needed.
    """

    def __init__(self, urdf_string: str | None = None, **kwargs: Any) -> None:
        super().__init__()
        variant = str(kwargs.pop("variant", "vega_1.vega_1_f5d6"))
        freeze_wheels = bool(kwargs.pop("freeze_wheels", True))
        freeze_hands = bool(kwargs.pop("freeze_hands", True))
        self.L_ee = str(kwargs.pop("left_ee_link", "L_ee"))
        self.R_ee = str(kwargs.pop("right_ee_link", "R_ee"))
        self.head_link_name = str(kwargs.pop("head_link", "head_l3"))

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        self._variant = variant
        self._freeze_wheels = freeze_wheels
        self._freeze_hands = freeze_hands
        self._frozen_joint_names: list[str] = []

        urdf = self._load_urdf(urdf_string)

        if self._freeze_wheels:
            for jn in _WHEEL_JOINTS:
                if jn in urdf.joint_map:
                    urdf.joint_map[jn].type = "fixed"
                    self._frozen_joint_names.append(jn)

        if self._freeze_hands:
            for jn in list(urdf.joint_map.keys()):
                if any(jn.startswith(prefix) for prefix in _HAND_JOINT_PREFIXES):
                    urdf.joint_map[jn].type = "fixed"
                    self._frozen_joint_names.append(jn)

        _strip_mimics_to_fixed_joints(urdf)
        urdf._update_actuated_joints()
        self._frozen_joint_names = sorted(set(self._frozen_joint_names))

        self.robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        names = self.robot.links.names
        for link, label in (
            (self.L_ee, "left EE"),
            (self.R_ee, "right EE"),
            (self.head_link_name, "head"),
        ):
            if link not in names:
                raise ValueError(
                    f"Link {link!r} ({label}) not found in URDF variant {self._variant!r}. "
                    f"Available links include similar names: "
                    f"{[n for n in names if 'ee' in n.lower() or 'head' in n.lower()][:20]}"
                )

        self.L_ee_link_idx = names.index(self.L_ee)
        self.R_ee_link_idx = names.index(self.R_ee)
        self.head_link_idx = names.index(self.head_link_name)

    def _load_default_urdf(self) -> yourdfpy.URDF:
        path_inside_repo = _dexmate_urdf_repo_path(self._variant)
        repo_root = ram.get_repo(repo_url=_DEXMATE_URDF_REPO_URL)
        self.urdf_path = str(
            ram.get_resource(
                repo_url=_DEXMATE_URDF_REPO_URL,
                path_inside_repo=path_inside_repo,
                resolve_packages=True,
            )
        )
        self.mesh_path = str(repo_root)
        return yourdfpy.URDF.load(self.urdf_path)

    @property
    @override
    def model_scale(self) -> float:
        return 0.5

    @property
    @override
    def orientation(self) -> jaxlie.SO3:
        return jaxlie.SO3.identity()

    @property
    @override
    def joint_var_cls(self) -> Any:
        return self.robot.joint_var_cls

    @property
    @override
    def actuated_joint_names(self) -> list[str]:
        return list(self.robot.joints.actuated_names)

    @property
    def frozen_joint_names(self) -> list[str]:
        return list(self._frozen_joint_names)

    @property
    @override
    def default_speed_ratio(self) -> float:
        return 1.2

    @override
    def forward_kinematics(self, config: jax.Array) -> dict[str, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(config)
        return {
            "left": jaxlie.SE3(fk[self.L_ee_link_idx]),
            "right": jaxlie.SE3(fk[self.R_ee_link_idx]),
            "head": jaxlie.SE3(fk[self.head_link_idx]),
        }

    @override
    def get_default_config(self) -> jax.Array:
        return jnp.array(
            [_VEGA_STANDING_POSE.get(name, 0.0) for name in self.actuated_joint_names]
        )

    @override
    def build_costs(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        target_Head: jaxlie.SE3 | None,
        q_current: jnp.ndarray | None = None,
    ) -> list[Cost]:
        costs: list[Cost] = []
        JointVar = self.robot.joint_var_cls
        joint_names = self.actuated_joint_names
        energy_weights = jnp.array(
            [_vega_rest_energy_weight(name) for name in joint_names]
        )
        centering_weights = jnp.array(
            [_vega_centering_weight(name) for name in joint_names]
        )
        standing_pose = self.get_default_config()

        if q_current is not None:
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=energy_weights,
                )
            )

        costs.append(
            pk.costs.rest_cost(
                JointVar(0),
                rest_pose=standing_pose,
                weight=centering_weights,
            )
        )

        costs.append(
            pk.costs.manipulability_cost(
                self.robot,
                JointVar(0),
                jnp.array([self.L_ee_link_idx, self.R_ee_link_idx], dtype=jnp.int32),
                weight=0.005,
            )
        )

        if target_L is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_L,
                    jnp.array(self.L_ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        if target_R is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_R,
                    jnp.array(self.R_ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        costs.append(
            pk.costs.limit_cost(
                self.robot,
                JointVar(0),
                weight=100.0,
            )
        )

        if target_Head is not None:
            costs.append(
                pk.costs.pose_cost(
                    robot=self.robot,
                    joint_var=JointVar(0),
                    target_pose=target_Head,
                    target_link_index=jnp.array(self.head_link_idx, dtype=jnp.int32),
                    pos_weight=0.0,
                    ori_weight=jnp.array([0.0, 0.0, 20.0]),
                )
            )

        costs.append(
            pk.costs.self_collision_cost(
                self.robot,
                self.robot_coll,
                JointVar(0),
                margin=0.05,
                weight=10.0,
            )
        )

        return costs
