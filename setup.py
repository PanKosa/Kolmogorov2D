from distutils.core import setup, Extension
from Cython.Build import cythonize

setup( ext_modules = cythonize(Extension(
            "SORy",
            sources=["SORy.pyx"],
            extra_compile_args=["-std=c++11"],
            language="c++"
     )))

