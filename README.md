# paramz

[![pypi](https://badge.fury.io/py/paramz.svg)](https://pypi.python.org/pypi/paramz)
[![build](https://travis-ci.org/sods/paramz.svg?branch=master)](https://travis-ci.org/sods/paramz)
[![codecov](https://codecov.io/github/sods/paramz/coverage.svg?branch=master)](https://codecov.io/github/sods/paramz?branch=master)

#### Coverage
![codecov.io](https://codecov.io/github/sods/paramz/branch.svg?branch=master)

Parameterization Framework for parameterized model creation and handling.
This is a lightweight framework for using parameterized models.

See examples model in 'paramz.examples.<tab>'

### Features:

 - Easy model creation with parameters
 - Fast optimized access of parameters for optimization routines
 - Memory efficient storage of parameters (only one copy in memory)
 - Renaming of parameters
 - Intuitive printing of models and parameters
 - Gradient saving directly inside parameters
 - Gradient checking of parameters
 - Optimization of parameters
 - Jupyter notebook integration
 - Efficient storage of models, for reloading
 - Efficient caching included

## Installation

You can install this package via pip

  pip install paramz

There is regular update for this package, so make sure to keep up to date
(Rerunning the install above will update the package and dependencies).

## Supported Platforms:

Python 2.7, 3.3 and higher

## Running unit tests:

Ensure nose is installed via pip:

    pip install nose

Run nosetests from the root directory of the repository:

    nosetests -v paramz/tests

or using setuptools

    python setup.py test

## Developer Documentation:

http://pythonhosted.org/paramz/

### Compiling documentation:

The documentation is stored in doc/ and is compiled with the Sphinx Python documentation generator, and is written in the reStructuredText format.

The Sphinx documentation is available here: http://sphinx-doc.org/latest/contents.html

**Installing dependencies:**

To compile the documentation, first ensure that Sphinx is installed. On Debian-based systems, this can be achieved as follows:

    sudo apt-get install python-pip
    sudo pip install sphinx

**Compiling documentation:**

The documentation can be compiled as follows:

    cd doc
    sphinx-apidoc -o source/ ../GPy/
    make html

The HTML files are then stored in doc/build/html

## Funding Acknowledgements

Current support for the paramz software is coming through the following projects.

* [EU FP7-PEOPLE Project Ref 316861](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/mlpm/) "MLPM2012: Machine Learning for Personalized Medicine"

