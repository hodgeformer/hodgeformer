#!/usr/bin/env python3
"""
Script to color meshes based on segmentation labels from JSON file using PyMeshLab.
The JSON file contains a dictionary mapping mesh file paths to segmentation arrays.
"""

import json
import numpy as np
import pymeshlab
from pathlib import Path
import argparse


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


def color_mesh_by_segmentation(mesh_path, segmentation, output_path):
    """
    Color a mesh based on face segmentation labels.
    Writes face colors directly to PLY file in ASCII format.
    
    Args:
        mesh_path: Path to input mesh file
        segmentation: Numpy array of shape (num_faces,) with segment labels
        output_path: Path to output colored mesh
    """
    # Load mesh with pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    
    # Get the current mesh
    mesh = ms.current_mesh()
    num_faces = mesh.face_number()
    num_vertices = mesh.vertex_number()
    
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


def process_json(json_path, base_dir, output_dir):
    """
    Process all meshes in the JSON file.
    
    Args:
        json_path: Path to JSON file containing mesh paths and segmentations
        base_dir: Base directory for resolving relative paths
        output_dir: Directory where colored meshes will be saved
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
            color_mesh_by_segmentation(mesh_path, segmentation, output_path)
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping...\n")
            continue
    
    print("All meshes processed!")


def print_help():
    """Print detailed help information."""
    help_text = """
=== Mesh Coloring Visualization Tool ===

This script colors meshes based on face segmentation labels from a JSON file.

USAGE:
    python visualize.py --json_path <path_to_json> --output_dir <output_directory>

REQUIRED ARGUMENTS:
    --json_path     Path to JSON file containing mesh paths and segmentation arrays
                    The JSON should map file paths to arrays of segmentation labels
    
    --output_dir    Directory where colored meshes will be saved
                    Will be created if it doesn't exist
                    Output files will have '_colored' suffix

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

EXAMPLE:
    python visualize.py --json_path out.json --output_dir ./results

FEATURES:
    - Generates distinct colors for each segment using HSV color space
    - Outputs PLY files in ASCII format with face colors
    - Automatically creates output directory if needed
    - Skips meshes that cannot be found or processed
    - Validates that segmentation array length matches face count

OUTPUT:
    Colored meshes are saved as PLY files with face color properties:
    - Each face gets an RGB color based on its segmentation label
    - Files are named: <original_name>_colored.ply
    - Format: ASCII PLY with vertex positions and face colors (R, G, B)
"""
    print(help_text)


def main():
    parser = argparse.ArgumentParser(
        description="Color meshes based on segmentation labels from JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize.py --json_path out.json --output_dir ./results
  python visualize.py --json_path /path/to/predictions.json --output_dir /path/to/colored_meshes

For more detailed help, run: python visualize.py --help
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
    
    # Base directory is root (for absolute paths in JSON)
    base_dir = Path("")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Mesh Coloring from JSON ===")
    print(f"JSON file: {json_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output suffix: '{OUTPUT_SUFFIX}'")
    print()
    
    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}")
        return
    
    # Process all meshes in JSON
    process_json(json_path, base_dir, output_dir)


if __name__ == "__main__":
    main()

