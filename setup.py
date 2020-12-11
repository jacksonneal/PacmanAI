from setuptools import setup, Extension
import sys
from Cython.Build import cythonize

if sys.platform == "win32":
    setup(ext_modules=cythonize(Extension(name="*", sources=["*.pyx"], extra_compile_args=["/std:c++17", "/O2"])))
else:
    setup(ext_modules=cythonize(Extension(name="*", sources=["*.pyx"], extra_compile_args=["std=c++17", "-O2"])))