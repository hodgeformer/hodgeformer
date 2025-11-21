import argparse

from mesh_sim.mesh import MeshSignedIncidenceMatrices
from mesh_sim.utils import read_obj, read_off, read_ply
from mesh_sim.utils_ml import extract_mesh_features


def parse_args():
    """
    Command line arguments for the `malware_bazaar_data/etl` script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract features from an input mesh with geometry information "
            "to be used in downstream machine learning pipeline."
        )
    )

    parser.add_argument(
        "--load_path",
        action="store",
        dest="load_path",
        help="Define path to `.obj` file with mesh geometry information",
    )

    parser.add_argument(
        "--save_path",
        action="store",
        dest="save_path",
        help="Define path for saving mesh patches as a `pkl` file(?).",
    )

    args = parser.parse_args()

    return args


def main(args):
    """
    Build a dataset with `apt` packages.
    """
    load_path = args.load_path

    if load_path.endswith(".obj"):
        loader = read_obj

    elif load_path.endswith(".off"):
        loader = read_off

    elif load_path.endswith(".ply"):
        loader = read_ply

    vertices, faces = loader(load_path)

    mesh = MeshSignedIncidenceMatrices(vertices, faces, uid="dummy")

    mesh._cache_geometrical_attrs()

    x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx = extract_mesh_features(
        mesh, mode="neighbors"
    )

    print("ok")


if __name__ == "__main__":
    args = parse_args()

    main(args)
