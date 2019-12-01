from setuptools import setup

setup(
    name='pytorchltr',
    version='0.1.0',
    description='Learning to Rank with PyTorch',
    url='https://github.com/rjagerman/pytorch',
    download_url = 'https://github.com/rjagerman/pytorch/archive/v0.1.0.tar.gz',
    author='Rolf Jagerman',
    author_email='rjagerman@gmail.com',
    license='MIT',
    packages=['pytorch',
              'pytorch.dataset',
              'test',
              'test.dataset'],
    install_requires=['numpy',
                      'scikit-learn',
                      'scipy',
                      'torch'],
    test_suite='nose.collector',
    tests_require=['nose']
)
