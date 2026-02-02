"""Setup configuration for CaRLeT package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements" / "base.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
else:
    requirements = []

setup(
    name="carlet",
    version="1.0.0",
    author="CaRLeT Research Team",
    author_email="",
    description="Causal Reinforcement Learning for Autonomous Driving Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imshujaa/Causal-RL-for-autonomous-driving-systems-",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "carlet=carlet.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
