import os
import sys

from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize
from distutils.extension import Extension

os.environ["CXX"] = "g++"
os.environ["CC"] = "gcc"

inc = [f"{os.getcwd()}/include", "include"]
sys.path.extend(inc)

setup(
    name="ecco",
    packages=find_packages(where="."),
    ext_modules=cythonize(
        [
            Extension(
                "ecco._ui",
                ["ecco/_ui.pyx"],
                include_path=inc,
                include_dirs=inc,
            ),
            Extension(
                "ecco.lrr.lts",
                ["ecco/lrr/lts.pyx"],
                include_path=inc,
                include_dirs=inc,
            ),
            Extension(
                "ecco.mrr.lts",
                ["ecco/mrr/lts.pyx"],
                include_path=inc,
                include_dirs=inc,
            ),
        ],
        language_level=3,
    ),
)
