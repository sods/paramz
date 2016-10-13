#!/usr/bin/env python
# -*- coding: utf-8 -*-

#===============================================================================
# Copyright (c) 2015, Max Zwiessele
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of paramax nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#===============================================================================

from __future__ import print_function
import os
from setuptools import setup
import codecs

def read(fname):
    with codecs.open(fname, 'r', 'latin') as f:
        return f.read()

#def read_to_rst(fname):
#    try:
#        import pypandoc
#        rstname = "{}.{}".format(os.path.splitext(fname)[0], 'rst')
#         pypandoc.convert(read(fname), 'rst', format='md', outputfile=rstname)
#         with open(rstname, 'r') as f:
#             rststr = f.read()
#         return rststr
#         #return read(rstname)
#     except ImportError:
#         return read(fname)

desc = read('README.rst')

version_dummy = {}
exec(read('paramz/__version__.py'), version_dummy)
__version__ = version_dummy['__version__']
del version_dummy

#Mac OS X Clang doesn't support OpenMP at the current time.
#This detects if we are building on a Mac
# def ismac():
#     return sys.platform[:6] == 'darwin'
#
# if ismac():
#     compile_flags = [ '-O3', ]
#     link_args = []
# else:
#     compile_flags = [ '-fopenmp', '-O3', ]
#     link_args = ['-lgomp']

# ext_mods = [Extension(name='GPy.kern._src.stationary_cython',
#                       sources=['GPy/kern/_src/stationary_cython.c',
#                                'GPy/kern/_src/stationary_utils.c'],
#                       include_dirs=[np.get_include(),'.'],
#                       extra_compile_args=compile_flags,
#                       extra_link_args = link_args),
#             Extension(name='GPy.util.choleskies_cython',
#                       sources=['GPy/util/choleskies_cython.c'],
#                       include_dirs=[np.get_include(),'.'],
#                       extra_link_args = link_args,
#                       extra_compile_args=compile_flags),
#             Extension(name='GPy.util.linalg_cython',
#                       sources=['GPy/util/linalg_cython.c'],
#                       include_dirs=[np.get_include(),'.'],
#                       extra_compile_args=compile_flags),
#             Extension(name='GPy.kern._src.coregionalize_cython',
#                       sources=['GPy/kern/_src/coregionalize_cython.c'],
#                       include_dirs=[np.get_include(),'.'],
#                       extra_compile_args=compile_flags)]

setup(name = 'paramz',
      version = __version__,
      author = 'Max Zwiessele',
      author_email = "ibinbei@gmail.com",
      description = ("The Parameterization Framework"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels",
      url = "https://github.com/sods/paramz",
      #ext_modules = ext_mods,
      packages = ["paramz",
                  "paramz.core",
                  "paramz.optimization",
                  "paramz.examples",
                  "paramz.tests"
                  ],
      #package_dir={'GPy': 'GPy'},
      #package_data = {'GPy': ['defaults.cfg', 'installation.cfg',
      #                        'util/data_resources.json',
      #                        'util/football_teams.json',
      #                        ]},
      #include_package_data = True,
      py_modules = ['paramz.__init__'],
      test_suite = 'paramz.tests',
      long_description=desc,
      install_requires=['numpy>=1.7', 'scipy', 'six', 'decorator>=4.0.10'],
      extras_require = {'docs':['sphinx'],
                        'notebook':['jupyter_client >= 4.0.6',
                                    'ipywidgets >= 4.0.3',
                                    'ipykernel >= 4.1.0',
                                    'notebook >= 4.0.5',
                                    ],
                        'optimization':['climin'],
                        },
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
