from setuptools import setup

setup(
    name='pytorchltr',
    version='0.1.0',
    description='Learning to Rank with PyTorch',
    url='https://github.com/rjagerman/pytorchltr',
    download_url = 'https://github.com/rjagerman/pytorchltr/archive/v0.1.0.tar.gz',
    author='Rolf Jagerman',
    author_email='rjagerman@gmail.com',
    license='MIT',
    packages=['pytorchltr',
              'pytorchltr.dataset',
              'pytorchltr.loss',
              'pytorchltr.evaluation',
              'test',
              'test.dataset',
              'test.evaluation'],
    install_requires=['numpy',
                      'scikit-learn',
                      'scipy',
                      'torch'],
    tests_require=['pytest']
)
