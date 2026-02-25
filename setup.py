from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mathematics-for-ai",
    version="0.2.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Educational repository and library for AI mathematics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mathematics-for-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "pre-commit>=2.0.0",
        ],
        "docs": [
            "mkdocs>=1.3.0",
            "mkdocs-material>=8.0.0",
            "mkdocstrings>=0.18.0",
            "pymdown-extensions>=9.0.0",
        ],
        "interactive": [
            "plotly>=5.0.0",
            "ipywidgets>=7.6.0",
            "jupyterlab>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "math-ai=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml"],
    },
)
