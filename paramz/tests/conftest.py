"""Compatibility base class for legacy unittest-style test cases.

Pytest collects ``unittest.TestCase`` subclasses natively, including their
``setUp`` methods. Keeping this named base avoids a broad test-only rename
while using the complete standard-library assertion implementation.
"""

import unittest


class AssertionsMixin(unittest.TestCase):
    pass
