import os
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

def readversion():
    ver = '1.2.0'
    with open('CHIMERA/version.py', 'w') as f:
        f.write('__version__="'+ver+'"')
        return ver

setup(
    name = "CHIMERA",
    version = readversion(),
    author = "Aoyan Dong",
    author_email = "sbia-software@uphs.upenn.edu",
    description = ("Clustering of heterogenous disease patterns within patient group."),
    license = "BSD",
    keywords = "CHIMERA heterogeneity clustering",
    url = "",
    packages = ['CHIMERA'],
    long_description = readme(),
    install_requires = ['numpy','sklearn'],
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Topic :: Machine Learning :: Clustering",
        "License :: BSD License",
        "Programming Language :: Python :: 2.7",
    ],
    zip_safe = False,
    scripts = ['bin/CHIMERA','bin/CHIMERA_TEST']
)
