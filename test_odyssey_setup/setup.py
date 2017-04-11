from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
	Extension("omp_test", 
		["omp_test.pyx"],
		libraries=["m"],
		extra_compile_args = ["-fopenmp"],
		extra_link_args=['-fopenmp']
		)
]
setup(
	name = "omp_test",
	cmdclass = {"build_ext": build_ext},
	ext_modules = ext_modules
)

# python setup.py build_ext --inplace
