from typing import List, Tuple

import numpy as np


def parse_vertex_row(row):
    """
    Parse vertex row.
    """
    try:
        vertices = row.strip().split()[1:]
    except IndexError:
        print("Row started with a `v` but did not hold any values")

    try:
        return [float(vertex) for vertex in vertices]
    except ValueError:
        print("Row vertex data were not valid, could not be converted to float.")

    return []


def parse_face_row(row):
    """
    Parse face row.
    """
    try:
        faces = row.strip().split()[1:]
    except IndexError:
        print("Row started with a `f` but did not hold any values")

    try:
        return [int(face.split("//")[0]) - 1 for face in faces]
    except ValueError:
        print("Row face data could not be parsed")

    return []


def read_obj(path: str) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Read `.obj` file.
    """
    vertices, faces = [], []

    with open(path, "r") as f:

        for row in f:

            if row.startswith("v "):
                vertices.append(parse_vertex_row(row))

            elif row.startswith("f "):
                faces.append(parse_face_row(row))

            else:
                continue

    return np.array(vertices), np.array(faces, dtype="int32")


def write_obj(vertices, faces, savepath):
    """
    Write `.obj` file given a GeometryOBJ class instance.
    """

    with open(savepath, "w") as f:
        try:
            for vtx in vertices:
                line = "v " + " ".join([str(v_i) for v_i in vtx])
                f.write(line + "\n")
        except Exception as e:
            print("Writing `.obj` file failed: {}".format(e))

        f.write("\n")
        try:
            for face in faces:
                line = "f " + " ".join((str(face + 1) for face in faces))
                f.write(line + "\n")
        except Exception as e:
            print("Writing `.obj` file failed: {}".format(e))

    return True


def validate_obj(vertices, faces):
    """ """
    pass
