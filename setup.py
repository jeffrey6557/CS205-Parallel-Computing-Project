from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy 

ext_modules = [
    Extension(
        "cost_function",
        ["cost_function.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='hello-parallel-world',
    ext_modules=cythonize(ext_modules),
)
