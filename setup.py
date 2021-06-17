from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np

extensions = []
setup(
    name="dynalearn",
    version=1.0,
    packages=find_packages(),
    include_package_data=True,
    ext_modules=extensions,
    setup_requires=[],
)
