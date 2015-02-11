from distutils.core import setup
from distutils.extension import Extension
import sys

from Cython.Build import cythonize
import numpy

setup(name='tste_cy',
      ext_modules = cythonize(Extension('cy_tste',["cy_tste.pyx",],
                                        include_dirs = [numpy.get_include()],
                                        extra_compile_args = [
                                            '-fopenmp',
                                            '-O3',
                                            '-ffast-math',
                                            #'-march=native',
                                            ],
                                        extra_link_args = ['-fopenmp'],
                  )),
      description="Cython flavor of TSTE",
)
