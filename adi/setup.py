# -*- coding: utf-8 -*-
"""Standalone setup for the adi-reader package.

From the repo root:    pip install -e adi/
Or from this dir:      pip install -e .

NOTE: This installs only the LabChart file reader. For hemodynamic
analysis, install the hemodynamics package separately.
"""

import os
import setuptools

# Resolve paths so this works whether invoked from repo root or adi/
here = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.dirname(here)

setuptools.setup(
    name="adi-reader",
    version="0.0.16a1",
    author="Jim Hokanson, irw-jh",
    author_email="",
    description="Reading LabChart recorded data",
    long_description="Python interface for reading .adicht (LabChart) files via the ADInstruments SDK.",
    long_description_content_type="text/plain",
    url="https://github.com/keller-aspire/adinstruments_sdk_python",
    packages=["adi"],
    package_dir={"adi": here},
    install_requires=["numpy", "cffi"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
