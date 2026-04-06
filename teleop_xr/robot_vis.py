import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from .config import RobotVisConfig


class RobotVisModule:
    """
    Module for serving robot visualization assets (URDF, meshes) and broadcasting state.
    """

    def __init__(self, app: FastAPI, config: RobotVisConfig):
        self.app = app
        self.config = config
        self.logger = logging.getLogger("teleop.robot_vis")
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/robot_assets/{file_path:path}")
        async def get_asset(file_path: str):
            self.logger.info(f"Asset request: {file_path}")
            full_path = ""

            def is_within(path: str, root: str) -> bool:
                try:
                    Path(path).resolve().relative_to(Path(root).resolve())
                    return True
                except ValueError:
                    return False

            if file_path == "robot.urdf":
                full_path = self.config.urdf_path
                # If we have a mesh path (repo root), try to rewrite absolute paths in URDF
                # to be relative, so the frontend can request them via /robot_assets/
                if self.config.mesh_path:
                    try:
                        with open(full_path, "r") as f:
                            content = f.read()

                        urdf_dir = Path(self.config.urdf_path).resolve().parent
                        mesh_root = Path(self.config.mesh_path).resolve()

                        def rewrite_mesh_filename(match: re.Match[str]) -> str:
                            quote = match.group(1)
                            raw = match.group(2)
                            raw_norm = raw.replace("\\", "/")

                            if "://" in raw_norm and not raw_norm.startswith(
                                "package://"
                            ):
                                return match.group(0)

                            if raw_norm.startswith("package://"):
                                return f"filename={quote}{raw_norm}{quote}"

                            path_obj = Path(raw_norm)
                            abs_path = (
                                path_obj.resolve()
                                if path_obj.is_absolute()
                                else (urdf_dir / path_obj).resolve()
                            )

                            try:
                                rel_path = abs_path.relative_to(mesh_root).as_posix()
                                return f"filename={quote}{rel_path}{quote}"
                            except ValueError:
                                return f"filename={quote}{raw_norm}{quote}"

                        new_content = re.sub(
                            r"filename\s*=\s*([\"'])([^\"']+)\1",
                            rewrite_mesh_filename,
                            content,
                        )

                        if new_content != content:
                            return Response(
                                content=new_content, media_type="application/xml"
                            )
                    except Exception as e:
                        self.logger.warning(f"Failed to rewrite URDF paths: {e}")
                        # Fallback to standard file serving

            elif "package://" in file_path:
                clean_path = file_path.split("package://")[-1]

                # Try ROS 2 resolution
                resolved = False
                try:
                    from ament_index_python.packages import get_package_share_directory

                    parts = clean_path.split("/")
                    pkg_name = parts[0]
                    rel_path = "/".join(parts[1:])
                    pkg_path = get_package_share_directory(pkg_name)
                    ros_path = os.path.join(pkg_path, rel_path)
                    if os.path.exists(ros_path):
                        full_path = ros_path
                        resolved = True
                except Exception:
                    pass

                if not resolved:
                    if self.config.mesh_path:
                        candidate = str(
                            (
                                Path(self.config.mesh_path).resolve() / clean_path
                            ).resolve()
                        )
                        if is_within(candidate, self.config.mesh_path):
                            full_path = candidate
                    else:
                        self.logger.warning(
                            f"Request for package resource '{file_path}' but 'mesh_path' is not configured."
                        )
                        full_path = clean_path
            else:
                # Try relative to URDF directory first
                urdf_dir = str(Path(self.config.urdf_path).resolve().parent)
                potential_paths = [
                    (str((Path(urdf_dir) / file_path).resolve()), urdf_dir)
                ]

                # If mesh_path is configured, try resolving against it
                if self.config.mesh_path:
                    potential_paths.append(
                        (
                            str(
                                (
                                    Path(self.config.mesh_path).resolve() / file_path
                                ).resolve()
                            ),
                            self.config.mesh_path,
                        )
                    )

                full_path = potential_paths[0][0]
                for p, root in potential_paths:
                    if is_within(p, root) and os.path.exists(p):
                        full_path = p
                        break

            if not os.path.exists(full_path):
                self.logger.warning(f"Asset not found: {full_path}")
                raise HTTPException(
                    status_code=404, detail=f"Asset not found: {file_path}"
                )

            media_type = None
            ext = os.path.splitext(full_path)[1].lower()
            if ext == ".stl":
                media_type = "application/octet-stream"
            elif ext == ".dae":
                media_type = "model/vnd.collada+xml"
            elif ext == ".obj":
                media_type = "text/plain"
            elif ext == ".urdf":
                media_type = "application/xml"
            elif ext == ".glb":
                media_type = "model/gltf-binary"
            elif ext == ".gltf":
                media_type = "model/gltf+json"

            return FileResponse(full_path, media_type=media_type)

    def get_frontend_config(self) -> Dict[str, Any]:
        return {
            "urdf_url": "/robot_assets/robot.urdf",
            "model_scale": self.config.model_scale,
            "initial_rotation_euler": self.config.initial_rotation_euler,
        }

    async def broadcast_state(self, connection_manager: Any, joints: Dict[str, float]):
        """
        Broadcasts the current joint state to all connected clients.

        Args:
            connection_manager: The ConnectionManager instance from Teleop class.
            joints: Dictionary mapping joint names to values (radians/meters).
        """
        message = {"type": "robot_state", "data": {"joints": joints}}
        await connection_manager.broadcast(json.dumps(message))
