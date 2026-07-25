"""
Setup script for PSCDL 2026 package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pscdl-2026",
    version="1.0.0",
    author="Jiyaro Joseph, Joseph S. Mathew",
    author_email="jiyarojoseph27.mec@gmail.com",
    description="Persistent Scene Change Detection for PSCDL 2026",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pscdl-2026-solution",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "pscdl=src.generate_mask:main",
        ],
    },
)