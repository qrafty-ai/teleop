import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from teleop_xr.robot_vis import RobotVisModule
from teleop_xr.config import RobotVisConfig


@pytest.fixture
def test_app():
    return FastAPI()


def test_urdf_path_rewriting(test_app, tmp_path):
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()

    urdf_content = f"""<robot name='test'>
      <link name='l'>
        <visual>
          <geometry>
            <mesh filename='{mesh_dir.as_posix()}/test.stl'/>
          </geometry>
        </visual>
      </link>
    </robot>"""
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(urdf_content)

    config = RobotVisConfig(urdf_path=str(urdf_path), mesh_path=str(mesh_dir))
    RobotVisModule(test_app, config)
    client = TestClient(test_app)

    response = client.get("/robot_assets/robot.urdf")
    assert response.status_code == 200
    assert mesh_dir.as_posix() not in response.text
    assert "test.stl" in response.text


def test_urdf_path_rewriting_fail(test_app, tmp_path, caplog):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot/>")
    config = RobotVisConfig(urdf_path=str(urdf_path), mesh_path=str(tmp_path))
    RobotVisModule(test_app, config)
    client = TestClient(test_app)

    with patch("teleop_xr.robot_vis.open", side_effect=Exception("rewrite error")):
        try:
            client.get("/robot_assets/robot.urdf")
        except Exception:
            pass
        assert "Failed to rewrite URDF paths: rewrite error" in caplog.text


def test_ros_package_resolution(test_app, tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot/>")

    config = RobotVisConfig(urdf_path=str(urdf_path), mesh_path=None)
    RobotVisModule(test_app, config)
    client = TestClient(test_app)

    mock_ament = MagicMock()
    mock_ament.get_package_share_directory.return_value = str(tmp_path / "pkg")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mesh.stl").write_bytes(b"content")

    with patch.dict(
        "sys.modules",
        {"ament_index_python": mock_ament, "ament_index_python.packages": mock_ament},
    ):
        response = client.get("/robot_assets/package://my_pkg/mesh.stl")
        assert response.status_code == 200
        assert response.content == b"content"


def test_ros_package_resolution_fail(test_app, tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot/>")
    config = RobotVisConfig(urdf_path=str(urdf_path), mesh_path=None)
    RobotVisModule(test_app, config)
    client = TestClient(test_app)

    response = client.get("/robot_assets/package://my_pkg/mesh.stl")
    assert response.status_code == 404


def test_asset_resolution_relative_to_urdf(test_app, tmp_path):
    urdf_dir = tmp_path / "robot_dir"
    urdf_dir.mkdir()
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text("<robot/>")

    asset_path = urdf_dir / "my_asset.glb"
    asset_path.write_bytes(b"glb_data")

    config = RobotVisConfig(urdf_path=str(urdf_path), mesh_path=None)
    RobotVisModule(test_app, config)
    client = TestClient(test_app)

    response = client.get("/robot_assets/my_asset.glb")
    assert response.status_code == 200
    assert response.headers["content-type"] == "model/gltf-binary"
    assert response.content == b"glb_data"


def test_urdf_rewrite_normalizes_parent_segments(test_app, tmp_path):
    repo_root = tmp_path / "repo"
    urdf_dir = repo_root / "robots" / "humanoid" / "vega_1"
    urdf_dir.mkdir(parents=True)

    mesh_rel = "../../hands/f5d6_hand/meshes/visual/th_l0.glb"
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text(
        f"""<robot name='vega'>
  <link name='l'>
    <visual><geometry><mesh filename='{mesh_rel}'/></geometry></visual>
  </link>
</robot>"""
    )

    mesh_path = repo_root / "robots" / "hands" / "f5d6_hand" / "meshes" / "visual"
    mesh_path.mkdir(parents=True)
    (mesh_path / "th_l0.glb").write_bytes(b"glb")

    config = RobotVisConfig(urdf_path=str(urdf_path), mesh_path=str(repo_root))
    RobotVisModule(test_app, config)
    client = TestClient(test_app)

    urdf_response = client.get("/robot_assets/robot.urdf")
    assert urdf_response.status_code == 200
    assert "../../" not in urdf_response.text
    rewritten = "robots/hands/f5d6_hand/meshes/visual/th_l0.glb"
    assert rewritten in urdf_response.text

    mesh_response = client.get(f"/robot_assets/{rewritten}")
    assert mesh_response.status_code == 200
    assert mesh_response.content == b"glb"


def test_package_traversal_rejected_with_mesh_root(test_app, tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot/>")
    mesh_root = tmp_path / "meshes"
    mesh_root.mkdir()
    (tmp_path / "escape.stl").write_bytes(b"escape")

    config = RobotVisConfig(urdf_path=str(urdf_path), mesh_path=str(mesh_root))
    RobotVisModule(test_app, config)
    client = TestClient(test_app)

    response = client.get("/robot_assets/package:%2F%2F..%2F..%2Fescape.stl")
    assert response.status_code == 404


def test_generic_traversal_rejected_with_mesh_root(test_app, tmp_path):
    urdf_dir = tmp_path / "urdf_dir"
    urdf_dir.mkdir()
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text("<robot/>")
    mesh_root = tmp_path / "meshes"
    mesh_root.mkdir()
    (tmp_path / "outside.glb").write_bytes(b"outside")

    config = RobotVisConfig(urdf_path=str(urdf_path), mesh_path=str(mesh_root))
    RobotVisModule(test_app, config)
    client = TestClient(test_app)

    response = client.get("/robot_assets/%2E%2E/%2E%2E/outside.glb")
    assert response.status_code == 404


def test_asset_not_found(test_app, tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot/>")
    config = RobotVisConfig(urdf_path=str(urdf_path), mesh_path=None)
    RobotVisModule(test_app, config)
    client = TestClient(test_app)

    response = client.get("/robot_assets/nonexistent.stl")
    assert response.status_code == 404


def test_all_mime_types(test_app, tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot/>")
    config = RobotVisConfig(urdf_path=str(urdf_path), mesh_path=None)
    RobotVisModule(test_app, config)
    client = TestClient(test_app)

    types = [
        ("test.dae", "model/vnd.collada+xml"),
        ("test.obj", "text/plain"),
        ("test.urdf", "application/xml"),
        ("test.glb", "model/gltf-binary"),
        ("test.gltf", "model/gltf+json"),
    ]
    for ext, mime in types:
        p = tmp_path / ext
        p.write_bytes(b"data")
        response = client.get(f"/robot_assets/{ext}")
        assert response.status_code == 200
        assert mime in response.headers["content-type"]


@pytest.mark.anyio
async def test_broadcast_state_direct(test_app, tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot/>")
    config = RobotVisConfig(urdf_path=str(urdf_path), mesh_path=None)
    module = RobotVisModule(test_app, config)

    manager = MagicMock()
    manager.broadcast = AsyncMock()

    await module.broadcast_state(manager, {"j1": 0.0})
    assert manager.broadcast.called


def test_get_frontend_config(test_app, tmp_path):
    config = RobotVisConfig(
        urdf_path=str(tmp_path / "robot.urdf"),
        mesh_path=None,
        model_scale=2.0,
        initial_rotation_euler=[1, 2, 3],
    )
    module = RobotVisModule(test_app, config)
    fe_config = module.get_frontend_config()
    assert fe_config["urdf_url"] == "/robot_assets/robot.urdf"
    assert fe_config["model_scale"] == 2.0
    assert fe_config["initial_rotation_euler"] == [1, 2, 3]


def test_package_resource_no_mesh_path_warning(test_app, tmp_path, caplog):
    config = RobotVisConfig(urdf_path=str(tmp_path / "robot.urdf"), mesh_path=None)
    RobotVisModule(test_app, config)
    client = TestClient(test_app)
    client.get("/robot_assets/package://foo/bar.stl")
    assert "Request for package resource" in caplog.text


def test_non_package_resource_not_found_warning(test_app, tmp_path, caplog):
    config = RobotVisConfig(urdf_path=str(tmp_path / "robot.urdf"), mesh_path=None)
    RobotVisModule(test_app, config)
    client = TestClient(test_app)
    client.get("/robot_assets/nonexistent.stl")
    assert "Asset not found" in caplog.text
