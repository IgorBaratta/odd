try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from os import path
import sys

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)


version = "0.0.1"

setup(name="odd",
      description="Optimized Domain Decomposition Library",
      long_description=long_description,
      version=version,
      author="Igor Baratta",
      author_email="baratta@ufmg.br",
      license="LGPL v3 or later",
      url='https://github.com/IgorBaratta/odd',
      zip_safe=False,
      packages=["odd"],
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 3 :: Only',
        ],

      install_requires=["numba", "mpi4py", "petsc4py"])
