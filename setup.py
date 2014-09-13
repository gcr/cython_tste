from distutils.core import setup
from distutils.extension import Extension
import sys

from Cython.Build import cythonize
import numpy

setup(name='tste_cy',
      ext_modules = cythonize(Extension('_tste',["_tste.pyx",],
                                        include_dirs = ["./yeppp-1.0.0/library/headers/",
                                                        numpy.get_include()],
                                        extra_compile_args = ['-fopenmp', '-O3', '-ffast-math', '-march=native'],
                                        extra_link_args = ['-fopenmp'],
                                        libraries = ['yeppp'],
                                        library_dirs = ['./yeppp-1.0.0/binaries/linux/x86_64/'],
                                        runtime_library_dirs = ['./yeppp-1.0.0/binaries/linux/x86_64/'],
                  )),
      description="Cython flavor of TSTE",
)
