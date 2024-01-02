import os
from setuptools import setup
from setuptools import find_packages
from setuptools import Extension

import numpy

try:
    from Cython.Build import cythonize
    CAN_CYTHONIZE = True
except ImportError:
    CAN_CYTHONIZE = False


def get_svmrank_parser_ext():
    """
    Gets the svmrank parser extension.

    This uses cython if possible when building from source, otherwise uses the
    packaged .c files to compile directly.
    """
    path = "pytorchltr/datasets/svmrank/parser"
    pyx_path = os.path.join(path, "svmrank_parser.pyx")
    c_path = os.path.join(path, "svmrank_parser.c")
    if CAN_CYTHONIZE and os.path.exists(pyx_path):
        return cythonize([Extension(
            "pytorchltr.datasets.svmrank.parser.svmrank_parser", [pyx_path],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])])
    else:
        return [Extension("pytorchltr.datasets.svmrank.parser.svmrank_parser",
                          [c_path])]


with open("README.md", "rt") as f:
    long_description = f.read()


setup(
    name="pytorchltr2",
    version="0.2.2",
    description="Learning to Rank with PyTorch (Fork of pytorchltr)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akreuzer/pytorchltr",
    author="Rolf Jagerman",
    author_email="a_kreuzer@posteo.de",
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*",)),
    python_requires='>=3.10',
    ext_modules=get_svmrank_parser_ext(),
    include_dirs=[numpy.get_include()],
    install_requires=["numpy",
                      "scikit-learn",
                      "scipy",
                      "torch"],
    tests_require=["pytest"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data = {
        "pytorchltr": ["py.typed"],
    },
)
