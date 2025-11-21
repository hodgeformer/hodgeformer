from setuptools import setup, find_namespace_packages

NAME = "mesh_opformer"
VERSION = "0.0.1"
DESCRIPTION = ""

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
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.4.0",
        "toml>=0.10",
        "einops>=0.8.0",
        "wandb",
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/cpu",
    ],
)
