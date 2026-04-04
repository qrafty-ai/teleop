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

from teleop_xr.ik.loader import (  # noqa: E402
    load_robot_class,
    reload_robot_class,
    list_available_robots,
    RobotLoadError,
)
from teleop_xr.ik.robot import BaseRobot  # noqa: E402
from teleop_xr.ik.robots.h1_2 import UnitreeH1Robot  # noqa: E402


def test_load_robot_class_none():
    """Test: load_robot_class(None) returns UnitreeH1Robot"""
    cls = load_robot_class(None)
    assert cls == UnitreeH1Robot


def test_load_robot_class_explicit():
    """Test: load_robot_class('teleop_xr.ik.robots.h1_2:UnitreeH1Robot') returns class"""
    cls = load_robot_class("teleop_xr.ik.robots.h1_2:UnitreeH1Robot")
    assert cls == UnitreeH1Robot


def test_load_robot_class_invalid_module():
    """Test: load_robot_class('nonexistent:Nope') raises RobotLoadError"""
    with pytest.raises(RobotLoadError) as excinfo:
        load_robot_class("nonexistent:Nope")
    assert "Failed to load robot class" in str(excinfo.value)


def test_load_robot_class_not_baserobot():
    """Test: load_robot_class('teleop_xr.ik.loader:RobotLoadError') raises (not a BaseRobot)"""
    with pytest.raises(RobotLoadError) as excinfo:
        load_robot_class("teleop_xr.ik.loader:RobotLoadError")
    assert "subclass of BaseRobot" in str(excinfo.value)


def test_list_available_robots():
    """Test: list_available_robots() returns dict with at least 'h1' entry"""
    # Note: This depends on Task 2 (entry points in pyproject.toml)
    # being active in the environment.
    available = list_available_robots()
    assert isinstance(available, dict)
    assert "h1" in available
    assert available["h1"] == "teleop_xr.ik.robots.h1_2:UnitreeH1Robot"


def test_load_robot_class_entry_point():
    """Test: load_robot_class('h1') returns UnitreeH1Robot via entry point"""
    # Note: This also depends on the entry point being registered.
    cls = load_robot_class("h1")
    assert cls == UnitreeH1Robot


def test_load_robot_class_entry_point_load_error(monkeypatch):
    """Test: load_robot_class raises RobotLoadError if entry point target import fails"""
    from unittest.mock import MagicMock

    # Mock entry point that resolves to a bad module path
    bad_ep = MagicMock()
    bad_ep.name = "bad_robot"
    bad_ep.value = "nonexistent:Nope"

    # Mock entry_points return value
    # Python 3.10+ returns a SelectableGroups, we need to mock access by name/iteration
    # entry_points(group=...) returns EntryPoints object (iterable)
    class MockEntryPoints:
        def __init__(self):
            self.names = ["bad_robot"]

        def __getitem__(self, name):
            if name == "bad_robot":
                return bad_ep
            raise KeyError(name)

        def __iter__(self):
            return iter([bad_ep])

    monkeypatch.setattr(
        "importlib.metadata.entry_points", lambda group: MockEntryPoints()
    )

    with pytest.raises(RobotLoadError) as excinfo:
        load_robot_class("bad_robot")
    assert "Failed to load robot class" in str(excinfo.value)
    assert "nonexistent" in str(excinfo.value)


def test_reload_robot_class_none():
    cls = reload_robot_class(None)
    assert issubclass(cls, BaseRobot)
    assert cls.__module__ == UnitreeH1Robot.__module__
    assert cls.__name__ == UnitreeH1Robot.__name__


def test_reload_robot_class_explicit():
    cls = reload_robot_class("teleop_xr.ik.robots.h1_2:UnitreeH1Robot")
    assert issubclass(cls, BaseRobot)
    assert cls.__module__ == UnitreeH1Robot.__module__
    assert cls.__name__ == UnitreeH1Robot.__name__


def test_list_available_robots_error(monkeypatch):
    """Test: list_available_robots handles exceptions gracefully"""

    def raise_err(*args, **kwargs):
        raise RuntimeError("Metadata broken")

    monkeypatch.setattr("importlib.metadata.entry_points", raise_err)

    # Should return empty dict instead of crashing
    available = list_available_robots()
    assert available == {}
