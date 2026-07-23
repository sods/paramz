"""Pytest support for the remaining legacy-style test methods."""

import pytest


class AssertionsMixin:
    def assertEqual(self, first, second, msg=None): assert first == second, msg
    def assertNotEqual(self, first, second, msg=None): assert first != second, msg
    def assertTrue(self, expression, msg=None): assert expression, msg
    def assertFalse(self, expression, msg=None): assert not expression, msg
    def assertIs(self, first, second, msg=None): assert first is second, msg
    def assertIsNot(self, first, second, msg=None): assert first is not second, msg
    def assertIsNone(self, value, msg=None): assert value is None, msg
    def assertIsInstance(self, value, expected_type, msg=None): assert isinstance(value, expected_type), msg
    def assertIn(self, member, container, msg=None): assert member in container, msg
    def assertNotIn(self, member, container, msg=None): assert member not in container, msg
    def assertListEqual(self, first, second, msg=None): assert list(first) == list(second), msg
    assertSequenceEqual = assertListEqual
    def assertSetEqual(self, first, second, msg=None): assert set(first) == set(second), msg
    def assertDictEqual(self, first, second, msg=None): assert dict(first) == dict(second), msg

    def assertRaises(self, exception, callable_=None, *args, **kwargs):
        if callable_ is None: return pytest.raises(exception)
        with pytest.raises(exception): callable_(*args, **kwargs)

    def assertRaisesRegex(self, exception, pattern, callable_=None, *args, **kwargs):
        if callable_ is None: return pytest.raises(exception, match=pattern)
        with pytest.raises(exception, match=pattern): callable_(*args, **kwargs)



@pytest.fixture(autouse=True)
def run_legacy_setup(request):
    instance = getattr(request, 'instance', None)
    setup = getattr(instance, 'setUp', None)
    if setup is not None: setup()
