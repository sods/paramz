# paramz

[![pypi](https://badge.fury.io/py/paramz.svg)](https://pypi.python.org/pypi/paramz)
[![build](https://travis-ci.org/sods/paramz.svg?branch=master)](https://travis-ci.org/sods/paramz)
[![codecov](https://codecov.io/github/sods/paramz/coverage.svg?branch=master)](https://codecov.io/github/sods/paramz?branch=master)
[![docStat](https://readthedocs.org/projects/paramz/badge/?version=latest)](http://paramz.readthedocs.org/en/latest/)

Parameterization Framework for parameterized model creation and handling.
This is a lightweight framework for using parameterized models.

See examples model in `paramz.examples.<tab>`

Features:

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

#### Coverage development
![codecov.io](https://codecov.io/github/sods/paramz/branch.svg?branch=master)
