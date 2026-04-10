# -*- coding: utf-8 -*-

import setuptools

setuptools.setup(
    name="hemodynamics",
    version="0.1.0",
    author="Mingfeng Li",
    author_email="",
    description="Hemodynamic signal analysis for LabChart data",
    long_description="Cardiac cycle detection, waveform features, spectral analysis, and PV loop processing.",
    long_description_content_type="text/plain",
    packages=["hemodynamics", "hemodynamics.tests"],
    package_dir={"hemodynamics": ".", "hemodynamics.tests": "tests"},
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
