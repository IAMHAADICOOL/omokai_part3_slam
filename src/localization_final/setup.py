from setuptools import find_packages, setup
import os
from glob import glob

package_name = "localization_final"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py") + glob("launch/*.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
        (
            os.path.join("share", package_name, "rviz"),
            glob("rviz/*.rviz"),
        ),
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="Team of 3",
    maintainer_email="student@university.edu",
    description="Graph SLAM with RANSAC polar line features from LaserScan",
    license="MIT",
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "task2_data_association_localization = localization_final.graph_slam:main",
            "task2_polar_line_extractor = localization_final.line_extractor:main",
            "arm_tuck = localization_final.arm_tuck:main",
        ],
    },
)