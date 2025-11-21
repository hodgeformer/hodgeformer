from setuptools import setup, find_namespace_packages

NAME = "mesh-sim"
VERSION = "0.0.1"
DESCRIPTION = "Mesh operations with Signed Incidence Matrices"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author="",
    author_email="",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    entry_points={"console_scripts": []},
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.10.0",
        "potpourri3d>=1.1.0",
        "open3d-cpu>=0.19.0",
        "matplotlib",
        "python-graphblas",
    ],
)
