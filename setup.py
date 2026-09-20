# Run "pip install -e ." to install the packages/modules I created
# in a way that allows me to edit the code and have the changes take effect
# immediately without having to reinstall

from setuptools import setup, find_packages
setup(
    name="DiffFluxReconstruction",
    packages=find_packages(),
    version="1.0.1",
    author="Rafael Parente",
    author_email="rafael.parente@tecnico.ulisboa.pt",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
    python_requires=">=3.14",
    install_requires=[
        # Core Math & Data Science
        "numpy>=2.0.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.8.0",
        "spacepy>=0.7.0",

        # Machine Learning & Optimization
        "torch>=2.0.0",
        "optuna>=3.5.0",
        "ray[tune]>=2.9.0",

        # UI & Utilities
        "pyqt5>=5.15.10",
    ],
)
