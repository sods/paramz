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

# Exact (module, name) pairs for pickle / NumPy reconstruction helpers.
# Module-wide allowances are avoided so imports hanging off support modules
# (e.g. codecs.builtins) and helpers such as numpy.testing.runstring cannot
# be resolved. See PR #61 review.
_EXACT_ALLOWED = frozenset({
    ("copyreg", "_reconstructor"),
    ("copy_reg", "_reconstructor"),
    ("collections", "OrderedDict"),
    ("codecs", "encode"),
    ("_codecs", "encode"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    # defaultdict factory pickled with IntArrayDict state
    ("paramz.core.lists_and_dicts", "intarray_default_factory"),
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

# Package prefixes whose *classes* may be resolved (not module-level functions).
DEFAULT_ALLOW_MODULE_PREFIXES = (
    "paramz",
    "GPy",
)


def _module_allowed(module, prefixes):
    for prefix in prefixes:
        if module == prefix or module.startswith(prefix + "."):
            return True
    return False


def _forbidden(module, name):
    return pickle.UnpicklingError(
        "global '{}.{}' is forbidden by paramz restricted unpickler".format(
            module, name
        )
    )


class RestrictedUnpickler(pickle.Unpickler):
    """
    ``pickle.Unpickler`` with an allow-list for GLOBAL opcodes.

    - Rejects dotted ``name`` values (protocol-4 attribute walks).
    - Allows an exact set of pickle/NumPy reconstructors.
    - Allows safe builtins by name.
    - For ``paramz`` / ``GPy`` / ``allow_modules`` prefixes, only accepts
      classes whose ``__module__`` lies under an allowed prefix.

    This blocks classic gadgets and the bypasses reported on PR #61. It is
    still not a full security boundary against hostile code in allow-listed
    classes; only load files from sources you trust.
    """

    def __init__(self, file, allow_modules=None, **kwargs):
        kwargs.setdefault("encoding", "latin1")
        super(RestrictedUnpickler, self).__init__(file, **kwargs)
        extra = tuple(allow_modules) if allow_modules else ()
        self._allow_modules = DEFAULT_ALLOW_MODULE_PREFIXES + extra

    def find_class(self, module, name):
        if "." in name:
            raise _forbidden(module, name)

        if module in ("builtins", "__builtin__"):
            if name not in _SAFE_BUILTINS:
                raise _forbidden(module, name)
            return super(RestrictedUnpickler, self).find_class(module, name)

        if (module, name) in _EXACT_ALLOWED:
            return super(RestrictedUnpickler, self).find_class(module, name)

        if _module_allowed(module, self._allow_modules):
            obj = super(RestrictedUnpickler, self).find_class(module, name)
            if isinstance(obj, type) and _module_allowed(obj.__module__, self._allow_modules):
                return obj

        raise _forbidden(module, name)


def restricted_load(file_obj, allow_modules=None):
    """Load from an open binary file object using :class:`RestrictedUnpickler`."""
    return RestrictedUnpickler(file_obj, allow_modules=allow_modules).load()
