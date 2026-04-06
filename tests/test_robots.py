import pytest

try:
    import jaxls  # noqa: F401
    import pyroki  # noqa: F401
    import jaxlie  # noqa: F401
    import yourdfpy  # noqa: F401
except ImportError:
    pytest.skip(
        "jaxls, pyroki, jaxlie, or yourdfpy not installed", allow_module_level=True
    )

import jax.numpy as jnp  # noqa: E402
import jaxlie  # noqa: E402
from unittest.mock import patch  # noqa: E402
from teleop_xr.ik.robot import BaseRobot  # noqa: E402
from teleop_xr.ik.robots.h1_2 import UnitreeH1Robot  # noqa: E402
from teleop_xr.ik.robots.vega import _dexmate_urdf_repo_path  # noqa: E402

H1_URDF = """
<robot name="h1_2">
  <link name="torso_link"/>
  <link name="L_hand_base_link"/>
  <link name="R_hand_base_link"/>
  <joint name="left_hip_yaw_joint" type="revolute"><parent link="torso_link"/><child link="l_leg"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <link name="l_leg"/>
  <joint name="j1" type="revolute"><parent link="torso_link"/><child link="L_hand_base_link"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="j2" type="revolute"><parent link="torso_link"/><child link="R_hand_base_link"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>
"""


class MockBaseRobot(BaseRobot):
    def _load_default_urdf(self):
        return None

    @property
    def actuated_joint_names(self):
        return ["j1"]

    @property
    def joint_var_cls(self):
        return None

    def get_vis_config(self):
        return None

    def forward_kinematics(self, config):
        return {"left": jaxlie.SE3.identity()}

    def get_default_config(self):
        return jnp.zeros(1)

    def build_costs(self, target_L, target_R, target_Head, q_current=None):
        return []


def test_base_robot_properties():
    robot = MockBaseRobot()
    assert robot.orientation.as_matrix().shape == (3, 3)
    assert robot.base_to_ros.as_matrix().shape == (3, 3)
    assert robot.ros_to_base.as_matrix().shape == (3, 3)
    assert robot.supported_frames == {"left", "right", "head"}
    assert robot.default_speed_ratio == 1.0


def test_h1_robot(tmp_path):
    dummy_urdf = tmp_path / "h1.urdf"
    dummy_urdf.write_text(H1_URDF)
    with patch("teleop_xr.ik.robots.h1_2.ram.get_resource") as mock_get:
        mock_get.return_value = dummy_urdf
        robot = UnitreeH1Robot()
        assert robot.supported_frames == {"left", "right", "head"}
        assert robot.default_speed_ratio == 1.2
        assert robot.joint_var_cls is not None
        q = robot.get_default_config()
        assert len(robot.actuated_joint_names) > 0
        fk = robot.forward_kinematics(q)
        assert "left" in fk
        costs = robot.build_costs(
            jaxlie.SE3.identity(),
            jaxlie.SE3.identity(),
            jaxlie.SE3.identity(),
            q_current=q,
        )
        assert len(costs) > 0
        assert robot.get_vis_config() is not None
        robot.urdf_path = ""
        assert robot.get_vis_config() is None


def test_vega_variant_to_ram_path_mapping():
    assert (
        _dexmate_urdf_repo_path("vega_1.vega_1_f5d6")
        == "robots/humanoid/vega_1/vega_1_f5d6.urdf"
    )
    assert (
        _dexmate_urdf_repo_path("vega_1u.vega_1u_f5d6")
        == "robots/humanoid/vega_1u/vega_1u_f5d6.urdf"
    )
