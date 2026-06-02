# Run "pip install -e ." to install the packages/modules I created
# in a way that allows me to edit the code and have the changes take effect
# immediately without having to reinstall

from setuptools import setup, find_packages
setup(
    name="DiffFluxReconstruction",
    packages=find_packages(),
    version="1.0.0",
    author="Rafael Parente",
    author_email="rafael.parente@tecnico.ulisboa.pt",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
    python_requires=">=3.14"
)
