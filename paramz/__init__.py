#===============================================================================
# Copyright (c) 2012 - 2014, GPy authors (see AUTHORS.txt).
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

from . import util
from .model import Model
from .parameterized import Parameterized
from .param import Param
from .core.observable_array import ObsAr
from paramz import transformations as constraints
from . import caching, optimization
from . import examples

from .__version__ import __version__

def _open_pickle_source(file_or_path):
    """
    Yield an open binary file for ``file_or_path``.

    If ``file_or_path`` is a path string, the file is closed after use.
    If it is already a file object, it is left open for the caller.
    """
    if isinstance(file_or_path, str):
        return open(file_or_path, 'rb'), True
    return file_or_path, False

def _unpickle_trusted(file_or_path):
    import pickle
    f, close = _open_pickle_source(file_or_path)
    try:
        return pickle.load(f, encoding='latin1')
    finally:
        if close:
            f.close()

def _unpickle_restricted(file_or_path, allow_modules=None):
    from .core.safe_pickle import restricted_load
    f, close = _open_pickle_source(file_or_path)
    try:
        return restricted_load(f, allow_modules=allow_modules)
    finally:
        if close:
            f.close()

def load(file_or_path, allow_modules=None, trusted=False):
    """
    Load a previously pickled model, using ``m.pickle('path/to/file.pickle')``.

    Only load pickles from sources you trust. By default a restricted
    unpickler is used so builtins gadgets such as ``exec`` cannot run during
    load (CWE-502 / issue #60). Prefix allow-listing is not a full security
    boundary against hostile code inside allow-listed packages.

    :param file_or_path: path to a pickle file, or an open binary file object
    :param allow_modules: optional extra module-name prefixes to allow when
        using the restricted loader (e.g. ``['mypkg']``)
    :param trusted: if True, use unrestricted ``pickle.load`` (unsafe; warns)
    """
    if trusted:
        import warnings
        warnings.warn(
            "paramz.load(..., trusted=True) uses unrestricted pickle.load; "
            "only use for files you trust (issue #60 / CWE-502).",
            UserWarning,
            stacklevel=2,
        )
        return _unpickle_trusted(file_or_path)
    return _unpickle_restricted(file_or_path, allow_modules=allow_modules)