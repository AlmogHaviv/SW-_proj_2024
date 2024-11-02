from setuptools import setup, Extension
import numpy as np

# List all the source files that are part of your extension
module = Extension('symnmf',
                   sources=['symnmfmodule.c', 'symnmf.c'])

setup(name='symnmf',
      version='1.0',
      description='Symmetric Non-negative Matrix Factorization implementation',
      ext_modules=[module])
