# pyright: reportCallIssue=false
"""
Dexmate Vega humanoid models (URDF from the dexmate-urdf package).

Install: pip install dexmate-urdf
See: https://github.com/dexmate-ai/dexmate-urdf

Design cross-check (URDF + costs):

**URDF 导入方式对比**
- **Unitree H1** (`h1_2.py`): `ram.get_resource` 拉 git 仓库里的 `.urdf`，在内存里把部分
  `joint_map[...].type = "fixed"`（如腿），再 `yourdfpy.URDF.load` → `pk.Robot.from_urdf`。
- **TeaArm** (`teaarm.py`): `ram.get_resource` + xacro；碰撞可用包内 `assets/teaarm/collision.json`
  做 `RobotCollision.from_sphere_decomposition`，否则 `from_urdf`。
- **Vega（本文件）**: 从已安装的 `dexmate_urdf.robots.humanoid` 取 `Path`（不经 RAM）；
  冻结轮/指 + `_strip_mimics_to_fixed_joints`（Dexmate 手指带 mimic，固定关节后必须清 mimic，
  否则 Pyroki 解析失败）；`mesh_path` 为 URDF 所在目录以便相对 mesh 路径解析。

**Cost 构建（与 H1 同套路，可对齐 TeaArm 调参）**
- 与 **H1** 一致：`rest_cost(q_current)` → `manipulability` → 双臂 `pose_cost_analytic_jac`
  → `limit_cost` → 头部 `pose_cost`（仅 z 轴朝向权重）→ 本实现含 **`self_collision_cost`**
  （接近 **OpenArm / TeaArm**；**H1** 仅用 `from_urdf` 碰撞对象但未在 `build_costs` 里加自碰撞项）。
- **TeaArm** 额外：`rest_cost` 用**按关节标量权重** `weight=jnp.array(...)`，以及第二段以零位为
  rest 的“回中”项；Vega 当前用标量 `weight=5.0` 与全零 `get_default_config`，若腰/躯干耦合差
  可仿 TeaArm 加 per-joint 权重或姿态偏置。
"""

from __future__ import annotations

import sys
from pathlib import Path
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


def _strip_mimics_to_fixed_joints(urdf: yourdfpy.URDF) -> None:
    """Mimic chains break if the driven joint is fixed; pyroki requires mimicked joints stay actuated."""
    fixed = {name for name, joint in urdf.joint_map.items() if joint.type == "fixed"}
    for joint in urdf.joint_map.values():
        if joint.mimic is not None and joint.mimic.joint in fixed:
            joint.mimic = None


def _dexmate_urdf_path(variant: str) -> Path:
    try:
        from dexmate_urdf import robots as dex_robots
    except ImportError as e:
        raise ImportError(
            "Dexmate Vega requires the 'dexmate-urdf' package. "
            "Install with: pip install dexmate-urdf"
        ) from e

    parts = variant.split(".")
    if len(parts) < 2:
        raise ValueError(
            "Dexmate variant must be a dotted path under dexmate_urdf.robots.humanoid, "
            f"e.g. 'vega_1.vega_1_f5d6', got {variant!r}"
        )

    mod: Any = dex_robots.humanoid
    for p in parts:
        mod = getattr(mod, p)
    urdf_obj = getattr(mod, "urdf", None)
    if urdf_obj is None:
        raise ValueError(f"No 'urdf' attribute on dexmate module for variant {variant!r}")
    return Path(urdf_obj)


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

        urdf = self._load_urdf(urdf_string)

        if self._freeze_wheels:
            for jn in _WHEEL_JOINTS:
                if jn in urdf.joint_map:
                    urdf.joint_map[jn].type = "fixed"

        if self._freeze_hands:
            for jn in list(urdf.joint_map.keys()):
                if any(jn.startswith(prefix) for prefix in _HAND_JOINT_PREFIXES):
                    urdf.joint_map[jn].type = "fixed"

        _strip_mimics_to_fixed_joints(urdf)
        urdf._update_actuated_joints()

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
        path = _dexmate_urdf_path(self._variant)
        self.urdf_path = str(path.resolve())
        self.mesh_path = str(path.parent.resolve())
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
        return jnp.zeros_like(self.robot.joints.lower_limits)

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

        if q_current is not None:
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=5.0,
                )
            )

        costs.append(
            pk.costs.manipulability_cost(
                self.robot,
                JointVar(0),
                jnp.array([self.L_ee_link_idx, self.R_ee_link_idx], dtype=jnp.int32),
                weight=0.01,
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
