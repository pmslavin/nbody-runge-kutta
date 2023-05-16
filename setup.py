from setuptools import setup, Extension

setup(
    name="rkfuncs",
    ext_modules=[Extension("rkfuncs", ["lib/rkfuncs.c"])]
)
