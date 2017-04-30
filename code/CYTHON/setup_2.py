from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy 
ext_modules = [
    Extension(
        "test_cost_function",
        ["test_cost_function.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='test_cost_function',
    ext_modules=cythonize(ext_modules),
)