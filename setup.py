from setuptools import setup, Extension

setup(
    name="rkfuncs",
    ext_modules=[Extension("rkfuncs",
        sources=["lib/rkfuncs.c"],
        extra_compile_args=["-Wall"],
    )]
)
