from distutils.core import setup, Extension

module = Extension("myModule", sources=["myModule.c"])

setup(name="PackageName", version="1.0", description="package for fib module", 
      ext_modules = [module])

