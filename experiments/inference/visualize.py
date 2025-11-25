#!/usr/bin/env python3
"""
Script to color meshes based on segmentation labels from JSON file.
The JSON file contains a dictionary mapping mesh file paths to segmentation arrays.
Supports both native Python and PyMeshLab implementations via --usepymeshlab flag.
"""

import json
import numpy as np
from pathlib import Path
import argparse
import sys


# Default values
OUTPUT_SUFFIX = "_colored"  # Will add this before file extension


def generate_colors(num_segments):
    """
    Generate distinct colors for each segment.
    
    Args:
        num_segments: Number of unique segments
        
    Returns:
        Dictionary mapping segment ID to RGB color (0-255 range)
    """
    np.random.seed(42)  # For reproducible colors
    colors = {}
    
    for i in range(num_segments):
        # Generate distinct colors using HSV to RGB conversion
        hue = (i * 137.508) % 360  # Golden angle for good distribution
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.8 + (i % 2) * 0.15
        
        # Convert HSV to RGB
        h = hue / 60.0
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if 0 <= h < 1:
            r, g, b = c, x, 0
        elif 1 <= h < 2:
            r, g, b = x, c, 0
        elif 2 <= h < 3:
            r, g, b = 0, c, x
        elif 3 <= h < 4:
            r, g, b = 0, x, c
        elif 4 <= h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        # Convert to 0-255 range
        colors[i] = (
            int((r + m) * 255),
            int((g + m) * 255),
            int((b + m) * 255)
        )
    
    return colors


# ============================================================================
# NATIVE PYTHON IMPLEMENTATION
# ============================================================================

def read_ply_mesh_native(ply_path):
    """
    Read a PLY mesh file using native Python.
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        tuple: (vertices, faces) where vertices is Nx3 array and faces is Mx3 array
    """
    vertices = []
    faces = []
    
    with open(ply_path, 'r') as f:
        # Read header
        line = f.readline().strip()
        if line != 'ply':
            raise ValueError(f"Not a valid PLY file: {ply_path}")
        
        # Parse header
        num_vertices = 0
        num_faces = 0
        in_header = True
        
        while in_header:
            line = f.readline().strip()
            
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                num_faces = int(line.split()[-1])
            elif line == 'end_header':
                in_header = False
        
        # Read vertices
        for _ in range(num_vertices):
            line = f.readline().strip()
            coords = line.split()
            vertices.append([float(coords[0]), float(coords[1]), float(coords[2])])
        
        # Read faces
        for _ in range(num_faces):
            line = f.readline().strip()
            parts = line.split()
            # First number is vertex count (should be 3 for triangles)
            vertex_count = int(parts[0])
            if vertex_count != 3:
                raise ValueError(f"Only triangle meshes are supported. Found face with {vertex_count} vertices.")
            face_vertices = [int(parts[1]), int(parts[2]), int(parts[3])]
            faces.append(face_vertices)
    
    return np.array(vertices), np.array(faces)


def write_ply_mesh_with_colors_native(output_path, vertices, faces, face_colors):
    """
    Write a PLY mesh file with face colors using native Python.
    
    Args:
        output_path: Path to output PLY file
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 array of face vertex indices
        face_colors: Mx3 array of face RGB colors (0-255)
    """
    num_vertices = len(vertices)
    num_faces = len(faces)
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        # Write faces with colors
        for face, color in zip(faces, face_colors):
            f.write(f"3 {face[0]} {face[1]} {face[2]} {color[0]} {color[1]} {color[2]}\n")


def color_mesh_native(mesh_path, segmentation, output_path):
    """
    Color a mesh using native Python implementation.
    
    Args:
        mesh_path: Path to input mesh file
        segmentation: Numpy array of shape (num_faces,) with segment labels
        output_path: Path to output colored mesh
    """
    # Read mesh using native Python
    vertices, faces = read_ply_mesh_native(mesh_path)
    
    num_faces = len(faces)
    num_vertices = len(vertices)
    
    print(f"  Vertices: {num_vertices}")
    print(f"  Faces in mesh: {num_faces}")
    print(f"  Segmentation length: {len(segmentation)}")
    
    # Validate segmentation array
    if len(segmentation) != num_faces:
        raise ValueError(
            f"Segmentation array length ({len(segmentation)}) does not match "
            f"number of faces ({num_faces})"
        )
    
    # Get unique segments and generate colors
    unique_segments = np.unique(segmentation)
    num_segments = len(unique_segments)
    print(f"  Unique segments: {num_segments} (IDs: {unique_segments})")
    
    # Generate color map
    color_map = generate_colors(num_segments)
    
    # Create a mapping from segment ID to color
    segment_to_color = {seg_id: color_map[i] for i, seg_id in enumerate(unique_segments)}
    
    # Create face color array
    face_colors = np.zeros((num_faces, 3), dtype=np.uint8)
    for face_idx, seg_id in enumerate(segmentation):
        color = segment_to_color[seg_id]
        face_colors[face_idx] = color
    
    # Write colored mesh
    write_ply_mesh_with_colors_native(output_path, vertices, faces, face_colors)
    
    print(f"  Saved: {output_path.name}")


# ============================================================================
# PYMESHLAB IMPLEMENTATION
# ============================================================================

def color_mesh_pymeshlab(mesh_path, segmentation, output_path):
    """
    Color a mesh using PyMeshLab implementation.
    
    Args:
        mesh_path: Path to input mesh file
        segmentation: Numpy array of shape (num_faces,) with segment labels
        output_path: Path to output colored mesh
    """
    try:
        import pymeshlab
    except ImportError:
        print("ERROR: PyMeshLab is not installed. Please install it with:")
        print("  pip install pymeshlab")
        sys.exit(1)
    
    # Load mesh with pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    
    # Get the current mesh
    mesh = ms.current_mesh()
    num_faces = mesh.face_number()
    num_vertices = mesh.vertex_number()
    
    print(f"  Vertices: {num_vertices}")
    print(f"  Faces in mesh: {num_faces}")
    print(f"  Segmentation length: {len(segmentation)}")
    
    # Validate segmentation array
    if len(segmentation) != num_faces:
        raise ValueError(
            f"Segmentation array length ({len(segmentation)}) does not match "
            f"number of faces ({num_faces})"
        )
    
    # Get unique segments and generate colors
    unique_segments = np.unique(segmentation)
    num_segments = len(unique_segments)
    print(f"  Unique segments: {num_segments} (IDs: {unique_segments})")
    
    # Generate color map
    color_map = generate_colors(num_segments)
    
    # Create a mapping from segment ID to color
    segment_to_color = {seg_id: color_map[i] for i, seg_id in enumerate(unique_segments)}
    
    # Get mesh data
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    
    # Write PLY file manually with face colors
    with open(output_path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        # Write faces with colors
        for face_idx, face in enumerate(faces):
            seg_id = segmentation[face_idx]
            color = segment_to_color[seg_id]
            f.write(f"3 {face[0]} {face[1]} {face[2]} {color[0]} {color[1]} {color[2]}\n")
    
    print(f"  Saved: {output_path.name}")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def color_mesh_by_segmentation(mesh_path, segmentation, output_path, use_pymeshlab):
    """
    Color a mesh based on face segmentation labels.
    
    Args:
        mesh_path: Path to input mesh file
        segmentation: Numpy array of shape (num_faces,) with segment labels
        output_path: Path to output colored mesh
        use_pymeshlab: If True, use PyMeshLab; otherwise use native Python
    """
    if use_pymeshlab:
        color_mesh_pymeshlab(mesh_path, segmentation, output_path)
    else:
        color_mesh_native(mesh_path, segmentation, output_path)


def process_json(json_path, base_dir, output_dir, use_pymeshlab):
    """
    Process all meshes in the JSON file.
    
    Args:
        json_path: Path to JSON file containing mesh paths and segmentations
        base_dir: Base directory for resolving relative paths
        output_dir: Directory where colored meshes will be saved
        use_pymeshlab: If True, use PyMeshLab; otherwise use native Python
    """
    # Load JSON
    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} meshes to process\n")
    
    # Process each mesh
    for i, (mesh_rel_path, segmentation_list) in enumerate(data.items(), 1):
        print(f"[{i}/{len(data)}] Processing: {mesh_rel_path}")
        
        # Resolve mesh path relative to the JSON file location
        mesh_path = base_dir / mesh_rel_path
        
        # Check if mesh exists
        if not mesh_path.exists():
            print(f"  WARNING: Mesh file not found: {mesh_path}")
            print(f"  Skipping...\n")
            continue
        
        # Convert segmentation list to numpy array
        segmentation = np.array(segmentation_list, dtype=np.int32)
        
        # Generate output path (in the script directory, with suffix)
        output_path = output_dir / f"{mesh_path.stem}{OUTPUT_SUFFIX}{mesh_path.suffix}"
        
        try:
            # Color the mesh
            color_mesh_by_segmentation(mesh_path, segmentation, output_path, use_pymeshlab)
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping...\n")
            continue
    
    print("All meshes processed!")


def print_help():
    """Print detailed help information."""
    help_text = """
=== Mesh Coloring Visualization Tool (v3 - Hybrid) ===

This script colors meshes based on face segmentation labels from a JSON file.
Supports both native Python and PyMeshLab implementations.

USAGE:
    python visualizev3.py --json_path <path_to_json> --output_dir <output_directory> [--usepymeshlab]

REQUIRED ARGUMENTS:
    --json_path     Path to JSON file containing mesh paths and segmentation arrays
                    The JSON should map file paths to arrays of segmentation labels
    
    --output_dir    Directory where colored meshes will be saved
                    Will be created if it doesn't exist
                    Output files will have '_colored' suffix

OPTIONAL ARGUMENTS:
    --usepymeshlab  Use PyMeshLab for mesh processing (default: native Python)
                    If specified, requires pymeshlab to be installed

JSON FILE STRUCTURE (out.json):
    The JSON file should be a dictionary mapping mesh file paths to segmentation arrays.
    Each segmentation array contains one integer label per face in the mesh.
    
    Structure:
    {
        "path/to/mesh1.ply": [0, 0, 1, 1, 2, 2, 3, 3, ...],
        "path/to/mesh2.ply": [1, 1, 1, 2, 2, 3, 4, 4, ...],
        ...
    }
    
    Example:
    {
        "../data/test/shrec__1.ply": [
            1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, ...
        ],
        "../data/test/shrec__2.ply": [
            0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, ...
        ]
    }
    
    Notes:
    - Keys: Absolute or relative paths to mesh files
    - Values: Arrays of integers (segmentation labels for each face)
    - Array length must match the number of faces in the mesh
    - Labels can be any non-negative integers (0, 1, 2, 3, ...)

EXAMPLES:
    # Using native Python (no dependencies except NumPy)
    python visualizev3.py --json_path out.json --output_dir ./results
    
    # Using PyMeshLab
    python visualizev3.py --json_path out.json --output_dir ./results --usepymeshlab

FEATURES:
    - Hybrid implementation: choose between native Python or PyMeshLab
    - Native Python: lightweight, no external mesh libraries
    - PyMeshLab: supports more mesh formats and operations
    - Generates distinct colors for each segment using HSV color space
    - Outputs PLY files in ASCII format with face colors
    - Automatically creates output directory if needed
    - Skips meshes that cannot be found or processed
    - Validates that segmentation array length matches face count

REQUIREMENTS:
    Native Python mode:
        - Python 3.6+
        - NumPy
    
    PyMeshLab mode:
        - Python 3.6+
        - NumPy
        - PyMeshLab (install with: pip install pymeshlab)

OUTPUT:
    Colored meshes are saved as PLY files with face color properties:
    - Each face gets an RGB color based on its segmentation label
    - Files are named: <original_name>_colored.ply
    - Format: ASCII PLY with vertex positions and face colors (R, G, B)

LIMITATIONS (Native Python mode):
    - Only supports ASCII PLY format (not binary)
    - Only supports triangle meshes
    - Does not preserve vertex normals, textures, or other attributes
"""
    print(help_text)


def main():
    parser = argparse.ArgumentParser(
        description="Color meshes based on segmentation labels from JSON file (Hybrid version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Native Python (default)
  python visualizev3.py --json_path out.json --output_dir ./results
  
  # Using PyMeshLab
  python visualizev3.py --json_path out.json --output_dir ./results --usepymeshlab

For more detailed help, run: python visualizev3.py --help
        """
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to JSON file containing mesh paths and segmentation arrays"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where colored meshes will be saved"
    )
    parser.add_argument(
        "--usepymeshlab",
        action="store_true",
        help="Use PyMeshLab for mesh processing (default: native Python)"
    )
    
    # Parse arguments and catch errors
    try:
        args = parser.parse_args()
    except SystemExit:
        # If parsing fails (missing required args), print custom help
        print("\n" + "="*60)
        print("ERROR: Missing required arguments!")
        print("="*60)
        print_help()
        raise
    
    # Convert to Path objects
    json_path = Path(args.json_path)
    output_dir = Path(args.output_dir)
    use_pymeshlab = args.usepymeshlab
    
    # Base directory is root (for absolute paths in JSON)
    base_dir = Path("")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which implementation to use
    implementation = "PyMeshLab" if use_pymeshlab else "Native Python"
    
    print("=== Mesh Coloring from JSON (v3 - Hybrid) ===")
    print(f"Implementation: {implementation}")
    print(f"JSON file: {json_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output suffix: '{OUTPUT_SUFFIX}'")
    print()
    
    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}")
        return
    
    # Process all meshes in JSON
    process_json(json_path, base_dir, output_dir, use_pymeshlab)


if __name__ == "__main__":
    main()

