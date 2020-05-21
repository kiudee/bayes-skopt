#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "scikit-optimize",
    "numpy",
    "scipy",
    "scikit-learn>=0.18.2,<0.23",
    "matplotlib",
    "emcee",
    "tqdm"
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Karlson Pfannschmidt",
    author_email="kiudee@mail.upb.de",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development"
    ],
    description="A fully Bayesian implementation of sequential model-based optimization",
    entry_points={"console_scripts": ["bask=bask.cli:main",],},
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="bask",
    name="bask",
    packages=find_packages(include=["bask", "bask.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/kiudee/bayes-skopt",
    version="0.5.0",
    zip_safe=False,
)
