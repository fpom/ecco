from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize
from distutils.extension import Extension

setup(
    name="ecco",
    packages=find_packages(where="."),
    ext_modules=cythonize(
        [
            Extension("ecco._ui", ["ecco/_ui.pyx"]),
            Extension("ecco.lrr.lts", ["ecco/lrr/lts.pyx"]),
            Extension("ecco.mrr.lts", ["ecco/mrr/lts.pyx"]),
        ],
        language_level=3,
    ),
)
