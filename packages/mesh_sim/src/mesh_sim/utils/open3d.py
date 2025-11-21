try:
    import open3d as o3d

except ImportError:
    print("`open3d` module is not found. `open3d` functionality cannot be used.")


def build_o3d_mesh(vertices, faces):
    """
    Create mesh with open3d.
    """
    v_s = o3d.utility.Vector3dVector(vertices)
    i_s = o3d.utility.Vector3iVector(faces)

    return o3d.geometry.TriangleMesh(v_s, i_s)
