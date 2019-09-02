try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import versioneer


setup(name='bayes-skopt',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Fully Bayesian optimization of costly and noisy target functions',
      license='Apache License 2.0',
      author='Karlson Pfannschmidt',
      packages=['bask'],
      install_requires=['scikit-optimize', 'numpy', 'scipy>=0.14.0',
                        'scikit-learn>=0.19.1', 'matplotlib'],
      )