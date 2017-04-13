from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "temp_code",
        ["temp_code.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    name='hello-parallel-world',
    ext_modules=cythonize(ext_modules),
)
