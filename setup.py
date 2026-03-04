"""
Setup configuration for Fruit & Vegetable Detection System.

This package provides a multi-model fusion detection system for identifying
groceries, vegetables, and fruits using YOLOv8 with ensemble learning.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="chefvision",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-time multi-model fusion detection system for identifying groceries, vegetables, and fruits using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darien-o/chefvision",
    project_urls={
        "Bug Reports": "https://github.com/darien-o/chefvision/issues",
        "Source": "https://github.com/darien-o/chefvision",
        "Documentation": "https://github.com/darien-o/chefvision/tree/main/docs",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "hypothesis>=6.82.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
        ],
        "datasets": [
            "roboflow>=1.1.0",
        ],
    },
    classifiers=[
        # Development status
        "Development Status :: 4 - Beta",
        
        # Intended audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating systems
        "Operating System :: OS Independent",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        
        # Other
        "Natural Language :: English",
        "Environment :: Console",
        "Typing :: Typed",
    ],
    keywords=[
        "yolo",
        "yolov8",
        "object-detection",
        "computer-vision",
        "machine-learning",
        "deep-learning",
        "ensemble-learning",
        "fusion",
        "fruit-detection",
        "vegetable-detection",
        "grocery-detection",
        "real-time-detection",
    ],
    license="MIT",
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "detect-fusion=detect_fusion:main",
            "download-models=download_models:main",
        ],
    },
)
