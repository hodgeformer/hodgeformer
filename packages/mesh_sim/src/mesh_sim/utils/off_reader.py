from typing import List, Tuple, TextIO

import numpy as np


def parse_vertices(file: TextIO, n_verts: int) -> List[List[float]]:
    """
    Parse face row.
    """
    return [
        [float(s) for s in file.readline().strip().split(" ")] for _ in range(n_verts)
    ]


def parse_faces(file: TextIO, n_faces: int) -> List[List[int]]:
    """
    Parse vertex row.
    """
    return [
        [int(s) for s in file.readline().strip().split(" ")][1:] for _ in range(n_faces)
    ]


def read_off(path: str) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Read `.off` file.
    """
    with open(path, "r") as f:

        if f.readline().strip() != "OFF":
            raise ValueError("Input file does not have a valid OFF header")

        n_verts, n_faces, _ = tuple([int(s) for s in f.readline().strip().split(" ")])

        vertices = parse_vertices(f, n_verts)

        faces = parse_faces(f, n_faces)

    return np.array(vertices), np.array(faces, dtype="int32")
