import pytest

try:
    import jaxlie  # noqa: F401
    import jaxls  # noqa: F401
    import pyroki  # noqa: F401
    import yourdfpy  # noqa: F401
except ImportError:
    pytest.skip("IK dependencies not installed", allow_module_level=True)

import sys
from unittest.mock import MagicMock

if "rclpy" not in sys.modules:
    mock_rclpy = MagicMock()

    class MockNode:
        def __init__(self, *args, **kwargs):
            self._parameters = {}

        def declare_parameter(self, name, value, descriptor=None):
            self._parameters[name] = value
            return MagicMock()

        def get_parameter(self, name):
            value = self._parameters.get(name)
            param = MagicMock()
            param.value = value
            param.get_parameter_value.return_value = MagicMock()
            if isinstance(value, bool):
                param.get_parameter_value.return_value.bool_value = value
            elif isinstance(value, int):
                param.get_parameter_value.return_value.integer_value = value
            elif isinstance(value, float):
                param.get_parameter_value.return_value.double_value = value
            elif isinstance(value, str):
                param.get_parameter_value.return_value.string_value = value
            return param

        def destroy_node(self):
            pass

        def get_logger(self):
            return MagicMock()

    mock_rclpy.node.Node = MockNode
    sys.modules["rclpy"] = mock_rclpy
    sys.modules["rclpy.node"] = mock_rclpy.node
    sys.modules["geometry_msgs.msg"] = MagicMock()
    sys.modules["sensor_msgs.msg"] = MagicMock()
    sys.modules["trajectory_msgs.msg"] = MagicMock()
    sys.modules["std_msgs.msg"] = MagicMock()
    sys.modules["tf2_ros"] = MagicMock()
    sys.modules["builtin_interfaces.msg"] = MagicMock()

from teleop_xr.ros2.__main__ import TeleopNode, Ros2CLI  # noqa: E402
from teleop_xr.config import InputMode


def test_teleop_node_params():
    cli = Ros2CLI()
    node = TeleopNode(cli)

    assert node.cli.mode == "teleop"
    assert node.cli.host == "0.0.0.0"
    assert node.cli.port == 4443
    assert node.cli.input_mode == InputMode.CONTROLLER
    assert node.cli.double_press_ms == 300
    assert node.cli.long_press_ms == 1000
    assert node.cli.head_topic is None
    assert node.cli.wrist_left_topic is None
    assert node.cli.wrist_right_topic is None
    assert node.cli.extra_streams == {}
    assert node.cli.frame_id == "xr_local"
    assert node.cli.publish_hand_tf is False
    assert node.cli.robot_class is None
    assert node.cli.robot_args_dict == {}
    assert node.cli.urdf_topic == "/robot_description"
    assert node.cli.urdf_timeout == 5.0
    assert node.cli.no_urdf_topic is False
    assert node.cli.ee_absolute_topic == "/teleop_xr/ee_absolute"

    node.destroy_node()


def test_teleop_node_param_override():
    cli = Ros2CLI(mode="ik")
    node = TeleopNode(cli)
    assert node.cli.mode == "ik"
    node.destroy_node()


def test_teleop_node_json_params():
    cli = Ros2CLI(
        extra_streams={"cam1": "/topic1"},
        robot_args='{"arg1": 123}',
    )
    node = TeleopNode(cli)

    assert node.cli.extra_streams == {"cam1": "/topic1"}
    assert node.cli.robot_args_dict == {"arg1": 123}

    cli.robot_args = "{invalid}"
    assert node.cli.robot_args_dict == {}

    node.destroy_node()
