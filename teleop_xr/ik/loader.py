import importlib
import importlib.metadata
from typing import cast
from teleop_xr.ik.robot import BaseRobot
from teleop_xr.ik.robots.h1_2 import UnitreeH1Robot


class RobotLoadError(Exception):
    """Custom exception for robot loading errors."""

    pass


def _require_robot_subclass(cls: object, source: str) -> type[BaseRobot]:
    if not isinstance(cls, type) or not issubclass(cls, BaseRobot):
        raise RobotLoadError(f"'{source}' is not a subclass of BaseRobot")
    return cls


def _resolve_robot_target(robot_spec: str | None) -> tuple[str, str]:
    if robot_spec is None:
        return UnitreeH1Robot.__module__, UnitreeH1Robot.__name__

    if ":" in robot_spec:
        module_name, class_name = robot_spec.split(":", 1)
        if not module_name or not class_name:
            raise RobotLoadError(
                f"Invalid robot specification: '{robot_spec}'. Must be in 'module:ClassName' format."
            )
        return module_name, class_name

    try:
        eps = importlib.metadata.entry_points(group="teleop_xr.robots")
    except Exception as e:
        raise RobotLoadError(f"Failed to discover robot entry points: {e}") from e

    if robot_spec not in eps.names:
        raise RobotLoadError(
            f"Invalid robot specification: '{robot_spec}'. Must be an entry point name or 'module:ClassName' format."
        )

    target = eps[robot_spec].value
    if ":" not in target:
        raise RobotLoadError(
            f"Entry point '{robot_spec}' has invalid target '{target}'. Expected 'module:ClassName'."
        )

    module_name, class_name = target.split(":", 1)
    return module_name, class_name


def load_robot_class(robot_spec: str | None = None) -> type[BaseRobot]:
    """
    Load a robot class based on the given specification.

    Precedence:
    1. If robot_spec is None, return UnitreeH1Robot.
    2. If robot_spec matches an entry point name in 'teleop_xr.robots', load that.
    3. If robot_spec contains ':', parse as 'module:ClassName'.
    4. Otherwise, raise RobotLoadError.

    Robot Constructor Contract:
    All robot classes must support the following constructor signature:
    `def __init__(self, urdf_string: str | None = None, **kwargs)`

    Args:
        robot_spec: The robot specification string or None.

    Returns:
        type[BaseRobot]: The loaded robot class.

    Raises:
        RobotLoadError: If the robot class cannot be loaded or is invalid.
    """
    module_name, class_name = _resolve_robot_target(robot_spec)
    source = robot_spec if robot_spec is not None else f"{module_name}:{class_name}"
    try:
        module = importlib.import_module(module_name)
        cls = cast(object, getattr(module, class_name))
    except (ImportError, AttributeError) as e:
        raise RobotLoadError(
            f"Failed to load robot class from spec '{source}': {e}"
        ) from e

    return _require_robot_subclass(cls, source)


def reload_robot_class(robot_spec: str | None = None) -> type[BaseRobot]:
    module_name, class_name = _resolve_robot_target(robot_spec)
    source = robot_spec if robot_spec is not None else f"{module_name}:{class_name}"
    try:
        module = importlib.import_module(module_name)
        reloaded_module = importlib.reload(module)
        cls = cast(object, getattr(reloaded_module, class_name))
    except (ImportError, AttributeError) as e:
        raise RobotLoadError(
            f"Failed to reload robot class from spec '{source}': {e}"
        ) from e

    return _require_robot_subclass(cls, source)


def list_available_robots() -> dict[str, str]:
    """
    List available robots via entry points without importing them.

    Returns:
        dict[str, str]: A mapping of robot names to their class paths.
    """
    try:
        eps = importlib.metadata.entry_points(group="teleop_xr.robots")
        return {ep.name: ep.value for ep in eps}
    except Exception:
        return {}
