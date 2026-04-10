# -*- coding: utf-8 -*-
"""Root setup — installs both adi-reader and hemodynamics packages.

For individual installs:
    pip install -e adi/
    pip install -e hemodynamics/

For everything at once:
    pip install -e .
"""

import setuptools

setuptools.setup(
    name="labchart-sdk",
    version="0.1.0",
    description="ADInstruments LabChart SDK + hemodynamic analysis toolkit",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "cffi",
        "scipy",
        "pandas",
        "h5py",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
