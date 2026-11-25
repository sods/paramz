"""Tests for restricted paramz.load (CIP-0005 / issue #60 / PR #61)."""
import io
import os
import pickle
import sys
import types
import warnings

import numpy as np
import pytest

import paramz
from paramz.param import Param
from paramz.parameterized import Parameterized
from paramz.transformations import Logexp


class _ExecPayload(object):
    def __reduce__(self):
        return (exec, ("raise RuntimeError('paramz-safe-pickle-pwned')",))


class _OsGetcwdPayload(object):
    def __reduce__(self):
        return (os.getcwd, ())


def _short_binunicode(text):
    raw = text.encode("utf-8")
    return b"\x8c" + bytes([len(raw)]) + raw


def test_default_load_rejects_exec_gadget():
    buf = io.BytesIO()
    pickle.dump(_ExecPayload(), buf)
    buf.seek(0)
    with pytest.raises(pickle.UnpicklingError, match="builtins.exec"):
        paramz.load(buf)


def test_default_load_rejects_os_gadget():
    buf = io.BytesIO()
    pickle.dump(_OsGetcwdPayload(), buf)
    buf.seek(0)
    # pickle may serialize os.getcwd as posix.getcwd / nt.getcwd.
    with pytest.raises(pickle.UnpicklingError, match=r"(os|posix|nt)\.getcwd"):
        paramz.load(buf)


def test_default_load_rejects_dotted_codecs_builtins_exec():
    # Protocol-4 dotted name walk: codecs.builtins.exec (PR #61 review).
    payload = (
        b"\x80\x04"
        + _short_binunicode("codecs")
        + _short_binunicode("builtins.exec")
        + b"\x93"
        + _short_binunicode("raise RuntimeError('dotted-exec')")
        + b"\x85R."
    )
    with pytest.raises(pickle.UnpicklingError, match=r"codecs\.builtins\.exec"):
        paramz.load(io.BytesIO(payload))


def test_default_load_rejects_numpy_testing_runstring():
    payload = (
        b"\x80\x04"
        + _short_binunicode("numpy.testing")
        + _short_binunicode("runstring")
        + b"\x93"
        + _short_binunicode("raise RuntimeError('runstring')")
        + b"}\x86R."
    )
    with pytest.raises(pickle.UnpicklingError, match=r"numpy\.testing\.runstring"):
        paramz.load(io.BytesIO(payload))


class _AllowModMarker(object):
    def __init__(self, value=1):
        self.value = value


def test_allow_modules_permits_extra_package_class():
    mod_name = "paramz_safe_pickle_allow_mod"
    mod = types.ModuleType(mod_name)
    marker_cls = type("Marker", (_AllowModMarker,), {})
    marker_cls.__module__ = mod_name
    mod.Marker = marker_cls
    sys.modules[mod_name] = mod
    try:
        buf = io.BytesIO()
        pickle.dump(marker_cls(7), buf, protocol=4)
        buf.seek(0)
        with pytest.raises(pickle.UnpicklingError):
            paramz.load(buf)
        buf.seek(0)
        restored = paramz.load(buf, allow_modules=[mod_name])
        assert type(restored).__name__ == "Marker"
        assert restored.value == 7
    finally:
        sys.modules.pop(mod_name, None)


def test_trusted_load_warns_and_runs_os_payload():
    buf = io.BytesIO()
    pickle.dump(_OsGetcwdPayload(), buf)
    buf.seek(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = paramz.load(buf, trusted=True)
    assert result == os.getcwd()
    assert any("trusted=True" in str(w.message) for w in caught)


def test_restricted_load_roundtrips_param():
    param = Param("test", np.arange(4.0).reshape(2, 2))
    param[0].constrain_positive()
    buf = io.BytesIO()
    pickle.dump(param, buf)
    buf.seek(0)
    restored = paramz.load(buf)
    assert restored.tolist() == param.tolist()


def test_restricted_load_roundtrips_parameterized():
    par = Parameterized("parameterized")
    child = Parameterized("rbf")
    child.lengthscale = Param("lengthscale", np.array([0.2, 0.3]), Logexp())
    child.link_parameter(child.lengthscale)
    par.kern = child
    par.link_parameter(par.kern)
    buf = io.BytesIO()
    pickle.dump(par, buf)
    buf.seek(0)
    restored = paramz.load(buf)
    np.testing.assert_array_equal(restored.param_array, par.param_array)
