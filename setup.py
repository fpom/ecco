import os

from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize
from distutils.extension import Extension

os.environ["CXX"] = "g++"
os.environ["CC"] = "gcc"

setup(
    name="ecco",
    packages=find_packages(where="."),
    ext_modules=cythonize(
        [
            Extension("ecco._ui", ["ecco/_ui.pyx"], include_dirs=["include"]),
            Extension("ecco.lrr.lts", ["ecco/lrr/lts.pyx"], include_dirs=["include"]),
            Extension("ecco.mrr.lts", ["ecco/mrr/lts.pyx"], include_dirs=["include"]),
        ],
        language_level=3,
    ),
)
