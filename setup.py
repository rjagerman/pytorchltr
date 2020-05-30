from setuptools import setup
from setuptools import find_packages


setup(
    name="pytorchltr",
    version="0.1.0",
    description="Learning to Rank with PyTorch",
    url="https://github.com/rjagerman/pytorchltr",
    author="Rolf Jagerman",
    author_email="rjagerman@gmail.com",
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*",)),
    install_requires=["numpy",
                      "scikit-learn",
                      "scipy",
                      "torch"],
    tests_require=["pytest"]
)
