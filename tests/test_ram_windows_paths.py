import pytest

try:
    import git  # noqa: F401
except ImportError:
    pytest.skip("git not installed", allow_module_level=True)

from pathlib import Path, PureWindowsPath  # noqa: E402
from unittest.mock import patch, MagicMock  # noqa: E402
import pytest  # noqa: E402
from teleop_xr import ram  # noqa: E402
from teleop_xr.robot_vis import RobotVisModule  # noqa: E402
from teleop_xr.config import RobotVisConfig  # noqa: E402
from fastapi import FastAPI  # noqa: E402


def test_replace_package_uris_uses_forward_slashes():
    """Mock _resolve_package to return a Windows-style path string, assert the resulting URDF content contains forward slashes."""
    urdf_content = '<mesh filename="package://my_pkg/mesh.stl"/>'
    repo_root = Path("/dummy/repo")

    # Mock _resolve_package to return a Windows path string
    with patch("teleop_xr.ram._resolve_package") as mock_resolve:
        mock_resolve.return_value = "C:\\Users\\test\\my_pkg"

        # We need to mock Path to return something that has .resolve().as_posix()
        # and correctly handles the Windows path string on Linux.
        with patch("teleop_xr.ram.Path") as MockPath:
            mock_pkg_path = MagicMock()
            MockPath.return_value = mock_pkg_path

            mock_full_path = MagicMock()
            mock_pkg_path.__truediv__.return_value = mock_full_path
            mock_full_path.resolve.return_value = mock_full_path
            mock_full_path.as_posix.return_value = "C:/Users/test/my_pkg/mesh.stl"

            resolved = ram._replace_package_uris(urdf_content, repo_root)

            assert "C:/Users/test/my_pkg/mesh.stl" in resolved
            assert "\\" not in resolved


def test_replace_dae_with_glb_uses_forward_slashes():
    """Mock _convert_dae_to_glb to return a PureWindowsPath, verify the URDF output uses forward slashes."""
    urdf_content = '<mesh filename="C:\\assets\\model.dae"/>'

    with patch("teleop_xr.ram._convert_dae_to_glb") as mock_convert:
        # Mock _convert_dae_to_glb to return a Windows Path object
        mock_convert.return_value = PureWindowsPath("C:\\assets\\model.glb")

        # We also need to mock Path to handle the Windows-style string correctly for the test
        with patch("teleop_xr.ram.Path") as MockPath:
            mock_path_instance = MagicMock()
            MockPath.return_value = mock_path_instance
            mock_path_instance.suffix.lower.return_value = ".dae"
            mock_path_instance.exists.return_value = True
            mock_path_instance.as_posix.return_value = "C:/assets/model.dae"

            result = ram._replace_dae_with_glb(urdf_content)

            assert 'filename="C:/assets/model.glb"' in result
            assert "\\" not in result


def test_mock_eval_find_returns_native_path():
    """Verify that _mock_eval_find does NOT convert to forward slashes (it should return native path for xacro)."""
    with patch("teleop_xr.ram._resolve_package") as mock_resolve:
        mock_resolve.return_value = "C:\\opt\\ros\\pkg"

        result = ram._mock_eval_find("pkg")
        assert result == "C:\\opt\\ros\\pkg"


def test_urdf_output_no_backslashes(tmp_path):
    """End-to-end test using process_xacro + _replace_package_uris."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    xacro_path = repo_root / "robot.xacro"
    xacro_path.write_text(
        '<robot name="test"><link name="base"><visual><geometry><mesh filename="package://pkg/mesh.stl"/></geometry></visual></link></robot>'
    )

    with patch("teleop_xr.ram._resolve_package") as mock_resolve:
        mock_resolve.return_value = "Z:\\ros_ws\\pkg"

        # Mock Path in ram.py to handle the Windows path correctly
        with patch("teleop_xr.ram.Path") as MockPath:
            # When Path(xacro_path) is called
            mock_xacro_path = MagicMock()
            mock_xacro_path.suffix = ".xacro"
            mock_xacro_path.name = "robot.xacro"

            # When Path(_resolve_package(...)) is called
            mock_pkg_path = MagicMock()

            def path_side_effect(arg):
                if arg == str(xacro_path):
                    return mock_xacro_path
                if arg == "Z:\\ros_ws\\pkg":
                    return mock_pkg_path
                return MagicMock()

            MockPath.side_effect = path_side_effect

            mock_full_path = MagicMock()
            mock_pkg_path.__truediv__.return_value = mock_full_path
            mock_full_path.resolve.return_value = mock_full_path
            mock_full_path.as_posix.return_value = "Z:/ros_ws/pkg/mesh.stl"

            # We also need to mock xacro.process_file to return a valid doc
            with patch("xacro.process_file") as mock_xacro_process:
                mock_doc = MagicMock()
                mock_xacro_process.return_value = mock_doc
                mock_doc.toprettyxml.return_value = (
                    '<mesh filename="package://pkg/mesh.stl"/>'
                )

                urdf_xml = ram.process_xacro(xacro_path, repo_root)

                assert "Z:/ros_ws/pkg/mesh.stl" in urdf_xml
                assert "\\" not in urdf_xml


@pytest.mark.anyio
async def test_robot_vis_path_stripping_with_forward_slash_urdf(tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    mesh_root = tmp_path / "work" / "repo"
    mesh_root.mkdir(parents=True)
    urdf_path.write_text(
        f'<mesh filename="{(mesh_root / "meshes" / "base.stl").as_posix()}"/>'
    )

    mesh_path = str(mesh_root)

    app = FastAPI()
    config = RobotVisConfig(urdf_path=str(urdf_path), mesh_path=mesh_path)

    RobotVisModule(app, config)

    handler = None
    for route in app.routes:
        if route.path == "/robot_assets/{file_path:path}":
            handler = route.endpoint
            break

    assert handler is not None

    response = await handler("robot.urdf")

    assert response.body.decode() == '<mesh filename="meshes/base.stl"/>'
    assert response.media_type == "application/xml"
