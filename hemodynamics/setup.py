# -*- coding: utf-8 -*-
"""Standalone setup for the hemodynamics analysis package.

From the repo root:    pip install -e hemodynamics/
Or from this dir:      pip install -e .
"""

import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

setuptools.setup(
    name="hemodynamics",
    version="0.1.0",
    author="Mingfeng Li",
    author_email="",
    description="Hemodynamic signal analysis for LabChart data",
    long_description="Cardiac cycle detection, waveform features, spectral analysis, and PV loop processing.",
    long_description_content_type="text/plain",
    packages=["hemodynamics", "hemodynamics.tests"],
    package_dir={
        "hemodynamics": here,
        "hemodynamics.tests": os.path.join(here, "tests"),
    },
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "h5py",
        "adi-reader",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.8",
)
