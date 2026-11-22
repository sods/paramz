"""Tests for restricted paramz.load (CIP-0005 / issue #60)."""
import io
import os
import pickle
import warnings

import numpy as np
import pytest

import paramz
from paramz.core.safe_pickle import is_allowed_global
from paramz.param import Param


class _ExecPayload(object):
    def __reduce__(self):
        return (exec, ("raise RuntimeError('paramz-safe-pickle-pwned')",))


class _OsGetcwdPayload(object):
    def __reduce__(self):
        return (os.getcwd, ())


def test_is_allowed_global_blocks_exec():
    assert is_allowed_global("builtins", "dict")
    assert not is_allowed_global("builtins", "exec")
    assert not is_allowed_global("builtins", "eval")
    assert not is_allowed_global("os", "system")
    assert is_allowed_global("paramz.param", "Param")
    assert is_allowed_global("numpy", "ndarray")
    assert is_allowed_global("GPy.kern", "RBF")
    assert not is_allowed_global("mypkg.models", "Foo")
    assert is_allowed_global("mypkg.models", "Foo", allow_modules=["mypkg"])


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


def test_allow_modules_permits_os_payload():
    buf = io.BytesIO()
    pickle.dump(_OsGetcwdPayload(), buf)
    buf.seek(0)
    result = paramz.load(buf, allow_modules=["os", "posix", "nt"])
    assert result == os.getcwd()


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
    buf = io.BytesIO()
    pickle.dump(param, buf)
    buf.seek(0)
    restored = paramz.load(buf)
    assert restored.tolist() == param.tolist()
