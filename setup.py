from setuptools import setup
from setuptools import find_packages


with open("README.md", "rt") as f:
    long_description = f.read()


setup(
    name="pytorchltr",
    version="0.1.0",
    description="Learning to Rank with PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rjagerman/pytorchltr",
    author="Rolf Jagerman",
    author_email="rjagerman@gmail.com",
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*",)),
    python_requires='>=3.6',
    install_requires=["numpy",
                      "scikit-learn",
                      "scipy",
                      "torch"],
    tests_require=["pytest"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
