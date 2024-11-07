from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
# List all the source files that are part of your extension
module = Extension('symnmf_module',
                   sources=['symnmfmodule.c', 'symnmf.c', 'symnmfhelpers.c'])

setup(name='symnmf_module',
      version='1.0',
      description='Symmetric Non-negative Matrix Factorization implementation',
      ext_modules=[module],
      cmdclass={"build_ext": build_ext}
)
