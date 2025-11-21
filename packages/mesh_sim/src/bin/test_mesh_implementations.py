import os
import argparse

import numpy as np

from mesh_sim.mesh import MeshSignedIncidenceMatrices
from mesh_sim.utils import (
    read_obj,
    read_off,
    read_ply,
)


def parse_args():
    """
    Command line arguments for the `malware_bazaar_data/etl` script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract patches from an input mesh with geometry information "
            "to be used in downstream machine learning pipeline."
        )
    )

    parser.add_argument(
        "--folder_path",
        action="store",
        dest="folder_path",
        help="Define path to folder with mesh files to test.",
    )

    args = parser.parse_args()

    return args


def build_o3d_mesh(vertices, faces):
    """
    Create mesh with open3d.
    """
    try:
        import open3d as o3d

    except ImportError:
        print("`open3d` module is not found. `None` is returned")
        return None

    v_s = o3d.utility.Vector3dVector(vertices)
    i_s = o3d.utility.Vector3iVector(faces)

    return o3d.geometry.TriangleMesh(v_s, i_s)


def test_normals(mesh, mesh_o3d):
    """
    Test mesh normals between different implementations
    """
    mesh_o3d.orient_triangles()

    mesh_o3d.compute_triangle_normals()

    ns_sim = mesh.face_normals()
    ns_o3d = np.asarray(mesh_o3d.triangle_normals)

    print(np.linalg.norm((ns_sim - ns_o3d), axis=1).max())


def test_normals_of_meshes_in_folder(folder_path: str) -> None:
    """
    Test mesh normals between different implementations.

    Parameters
    ----------
    folder_path : str
        Path to folder with mesh files to test.
    """
    for f in os.listdir(folder_path):

        load_path = os.path.join(folder_path, f)

        if load_path.endswith(".obj"):
            loader = read_obj

        elif load_path.endswith(".off"):
            loader = read_off

        elif load_path.endswith(".ply"):
            loader = read_ply

        vertices, faces = loader(load_path)

        mesh = MeshSignedIncidenceMatrices(vertices, faces, uid="dummy")

        mesh_o3d = build_o3d_mesh(vertices, faces)

        test_normals(mesh, mesh_o3d)

    print("ok")


if __name__ == "__main__":
    args = parse_args()

    test_normals_of_meshes_in_folder(args.folder_path)
