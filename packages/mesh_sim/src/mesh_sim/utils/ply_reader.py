import potpourri3d as pp3d


def read_ply(path):
    """
    Read `.ply` file.
    """
    vertices, faces = pp3d.read_mesh(path)

    return vertices.astype("float32"), faces.astype("int32")
