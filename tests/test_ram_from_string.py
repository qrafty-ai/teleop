import hashlib
import pytest

pytest.importorskip("git")
from teleop_xr import ram


def test_from_string_basic(tmp_path):
    urdf_content = '<robot name="test_robot"><link name="base_link"/></robot>'
    cache_dir = tmp_path / "cache"

    urdf_path, mesh_path = ram.from_string(urdf_content, cache_dir=cache_dir)

    content_hash = hashlib.sha256(urdf_content.encode()).hexdigest()[:12]
    expected_path = cache_dir / "processed" / f"string_{content_hash}.urdf"

    assert urdf_path == expected_path
    assert urdf_path.exists()
    assert urdf_path.read_text() == urdf_content
    assert mesh_path is None


def test_from_string_with_meshes(tmp_path):
    # Create some dummy mesh files
    mesh_dir = tmp_path / "my_robot" / "meshes"
    mesh_dir.mkdir(parents=True)
    mesh_file = mesh_dir / "base.stl"
    mesh_file.write_text("dummy stl")

    urdf_content = f"""<robot name="test_robot">
    <link name="base_link">
        <visual>
            <geometry>
                <mesh filename="{mesh_file}"/>
            </geometry>
        </visual>
    </link>
</robot>"""

    cache_dir = tmp_path / "cache"
    urdf_path, mesh_path = ram.from_string(urdf_content, cache_dir=cache_dir)

    assert urdf_path.exists()
    # For a single mesh, we expect the directory of that mesh.
    assert mesh_path is not None
    assert mesh_path.replace("\\", "/") == mesh_dir.as_posix()


def test_from_string_too_generic_prefix(tmp_path):
    # Test case where common path is /usr (too generic)
    urdf_content = """<robot name="test_robot">
    <link name="l1"><visual><geometry><mesh filename="/usr/share/m1.stl"/></geometry></visual></link>
    <link name="l2"><visual><geometry><mesh filename="/usr/lib/m2.stl"/></geometry></visual></link>
</robot>"""

    cache_dir = tmp_path / "cache"
    urdf_path, mesh_path = ram.from_string(urdf_content, cache_dir=cache_dir)
    assert mesh_path is None


def test_from_string_package_uri(tmp_path, monkeypatch):
    # Mock a package path
    pkg_path = tmp_path / "my_pkg"
    mesh_dir = pkg_path / "meshes"
    mesh_dir.mkdir(parents=True)
    mesh_file = mesh_dir / "link.stl"
    mesh_file.write_text("dummy")

    urdf_content = """<robot name="test_robot">
    <link name="base_link">
        <visual>
            <geometry>
                <mesh filename="package://my_pkg/meshes/link.stl"/>
            </geometry>
        </visual>
    </link>
</robot>"""

    def get_pkg_share(pkg):
        if pkg == "my_pkg":
            return str(pkg_path)
        return None

    monkeypatch.setattr(ram, "_get_ros_package_share_directory", get_pkg_share)

    cache_dir = tmp_path / "cache"
    urdf_path, mesh_path = ram.from_string(urdf_content, cache_dir=cache_dir)

    content = urdf_path.read_text()
    assert "package://" not in content
    assert mesh_file.as_posix() in content
    assert mesh_path is not None
    assert mesh_path.replace("\\", "/") == mesh_dir.as_posix()
