#!/usr/bin/env python
import setuptools
import os


os.chmod("examples/run.py", 0o744)
os.chmod("examples/run_MNIST.py", 0o744)
os.chmod("examples/run_distributed.py", 0o744)

setuptools.setup(
    name="hamiltonianNet",
    version="0.0.1",
    author="Clara Galimberti",
    author_email="clara.galimberti@epfl.ch",
    description="PyTorch package for Hamiltonian DNNs",
    url="https://github.com/ClaraGalimberti/hamiltonianNet",
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.4.0',
                      'numpy>=1.18.1',
                      'matplotlib>=3.1.3',
                      'torchvision>=0.5.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)
