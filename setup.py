import pathlib, inspect
from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize
from distutils.extension import Extension

import ecco

VERSION = '0.1'
long_description = pathlib.Path("README.md").read_text(encoding="utf-8")
description = inspect.cleandoc(ecco.__doc__).splitlines()[0]

setup(name="ecco",
      version=VERSION,
      description=description,
      long_description=long_description,
      url="https://github.com/fpom/ecco",
      author="Franck Pommereau",
      author_email="franck.pommereau@univ-evry.fr",
      classifiers=["Development Status :: 4 - Beta",
                   "Intended Audience :: Developers",
                   "Topic :: Scientific/Engineering",
                   "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
                   "Programming Language :: Python :: 3",
                   "Operating System :: OS Independent"],
      packages=find_packages(where="."),
      python_requires=">=3.7",
      ext_modules = cythonize([Extension("ecco._ui",
                                         ["ecco/_ui.pyx"])],
                              language_level=3),
)
