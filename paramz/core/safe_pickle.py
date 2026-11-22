#===============================================================================
# Copyright (c) 2026, paramz authors
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
"""Restricted pickle loading for :func:`paramz.load` (CWE-502 / issue #60)."""

from __future__ import annotations

import pickle

# Modules pickle itself uses when reconstructing ordinary containers / codecs.
_PICKLE_SUPPORT_MODULES = frozenset({
    "builtins",
    "__builtin__",
    "copyreg",
    "copy_reg",
    "collections",
    "collections.abc",
    "_compat_pickle",
    "codecs",
    "_codecs",
})

# Callables / types safe to resolve from builtins during unpickle.
_SAFE_BUILTINS = frozenset({
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "Exception",
    "LookupError",
    "NameError",
    "NoneType",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "bool",
    "bytearray",
    "bytes",
    "complex",
    "dict",
    "enumerate",
    "float",
    "frozenset",
    "int",
    "list",
    "memoryview",
    "object",
    "property",
    "range",
    "set",
    "slice",
    "staticmethod",
    "str",
    "tuple",
    "type",
})

# Default package prefixes for paramz / NumPy / GPy model graphs.
DEFAULT_ALLOW_MODULE_PREFIXES = (
    "paramz",
    "numpy",
    "GPy",
)


def _module_allowed(module, prefixes):
    for prefix in prefixes:
        if module == prefix or module.startswith(prefix + "."):
            return True
    return False


def is_allowed_global(module, name, allow_modules=None):
    """Return True if ``module.name`` may be resolved by the restricted unpickler."""
    prefixes = DEFAULT_ALLOW_MODULE_PREFIXES
    if allow_modules:
        prefixes = prefixes + tuple(allow_modules)

    if module in ("builtins", "__builtin__"):
        return name in _SAFE_BUILTINS

    if module in _PICKLE_SUPPORT_MODULES:
        return True

    return _module_allowed(module, prefixes)


class RestrictedUnpickler(pickle.Unpickler):
    """
    ``pickle.Unpickler`` that only resolves classes from an allow-list.

    This blocks classic builtins gadgets (e.g. ``exec``) used in crafted
    pickles. It is not a full security boundary against hostile code in
    allow-listed packages; only load files from sources you trust.
    """

    def __init__(self, file, allow_modules=None, **kwargs):
        kwargs.setdefault("encoding", "latin1")
        super(RestrictedUnpickler, self).__init__(file, **kwargs)
        extra = tuple(allow_modules) if allow_modules else ()
        self._allow_modules = DEFAULT_ALLOW_MODULE_PREFIXES + extra

    def find_class(self, module, name):
        if module in ("builtins", "__builtin__"):
            allowed = name in _SAFE_BUILTINS
        elif module in _PICKLE_SUPPORT_MODULES:
            allowed = True
        else:
            allowed = _module_allowed(module, self._allow_modules)
        if not allowed:
            raise pickle.UnpicklingError(
                "global '{}.{}' is forbidden by paramz restricted unpickler".format(
                    module, name
                )
            )
        return super(RestrictedUnpickler, self).find_class(module, name)


def restricted_load(file_obj, allow_modules=None):
    """Load from an open binary file object using :class:`RestrictedUnpickler`."""
    return RestrictedUnpickler(file_obj, allow_modules=allow_modules).load()
