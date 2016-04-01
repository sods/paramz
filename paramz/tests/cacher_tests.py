'''
Created on 4 Sep 2015

@author: maxz
'''
import unittest
from ..caching import Cacher
from pickle import PickleError
from ..core.observable_array import ObsAr
import numpy as np
from paramz.caching import Cache_this

class TestDecorator(unittest.TestCase):
    def setUp(self):
        opcalls = [0, 0, 0]
        class O(object):
            @Cache_this(ignore_args=[0, 3], force_kwargs=('force',))
            def __call__(self, x, y, ignore_this, force=False):
                """Documentation"""
                opcalls[0] += 1
                opcalls[1] = ignore_this
                if force is not False:
                    opcalls[2] = force
                return x + y
        self.opcalls = opcalls
        self.O = O
        self.cached = self.O()

    def test_cacher_cache(self):
        class O(object):
            def __init__(self):
                self._test = Cacher(self.test)
                self._test_other = Cacher(self.other_method)
            def test(self, x, y):
                return x * y
            def other_method(self, x, y):
                return x + y

        a = ObsAr(np.random.normal(0, 1, (2, 1)))
        b = ObsAr(np.random.normal(0, 1, (2, 1)))

        o = O()

        ab = o._test(a, b)
        self.assertIs(ab, o._test(a, b))
        aba = o._test(ab, a)
        self.assertIsNot(ab, aba)
        # ab still cached:
        self.assertIs(ab, o._test(a, b))
        # change a
        a[0] = 15
        self.assertIsNot(ab, o._test(a, b))
        self.assertIsNot(ab, o._test(ab, a))

        self.assertIs(o.cache[o.test], o._test)
        self.assertIs(o.cache[o.other_method], o._test_other)


    def test_opcalls(self):
        a = ObsAr(np.random.normal(0, 1, (2, 1)))
        b = ObsAr(np.random.normal(0, 1, (2, 1)))

        ab = self.cached(a, b, 'ignored')
        self.assertEqual(self.opcalls[0], 1)
        self.assertEqual(self.opcalls[1], 'ignored')
        self.assertIs(ab, self.cached(a, b, 2))
        self.assertEqual(self.opcalls[1], 'ignored')
        self.assertEqual(self.opcalls[0], 1)
        abnew = self.cached(a, b, 3, force='given')
        self.assertEqual(self.opcalls[0], 2)
        self.assertEqual(self.opcalls[1], 3)
        self.assertEqual(self.opcalls[2], 'given')
        self.assertIsNot(ab, abnew)
        np.testing.assert_array_equal(abnew, ab)
        abforced = self.cached(a, b, 4, force='given2')
        self.assertEqual(self.opcalls[0], 3)
        self.assertEqual(self.opcalls[1], 4)
        self.assertEqual(self.opcalls[2], 'given2')
        np.testing.assert_array_equal(abnew, abforced)

    def test_signature(self):
        try:
            from inspect import getfullargspec, getdoc
            self.assertEqual(getfullargspec(self.cached.__call__), getfullargspec(self.O.__call__))
            self.assertEqual(getdoc(self.cached.__call__), getdoc(self.O.__call__))
        except ImportError:
            try:
                from inspect import signature, getdoc
                print(signature(self.cached.__call__), signature(self.O.__call__))
                self.assertEqual(signature(self.cached.__call__), signature(self.O.__call__))
                self.assertEqual(getdoc(self.cached.__call__), getdoc(self.O.__call__))
            except ImportError:
                from inspect import getargspec, getdoc
                self.assertEqual(getargspec(self.cached.__call__), getargspec(self.O.__call__))
                self.assertEqual(getdoc(self.cached.__call__), getdoc(self.O.__call__))

    def test_offswitch(self):
        a = ObsAr(np.random.normal(0, 1, (2, 1)))
        b = ObsAr(np.random.normal(0, 1, (2, 1)))
        ab1 = self.cached(a, b, 'ignored')  # cached is ab
        ab2 = self.cached(a, b, 2)  # cached is still ab
        self.assertIs(ab1, ab2)  # thus ab and ab2 are the SAME objects
        self.cached.cache.disable_caching()  # cached is deleted
        ab3 = self.cached(a, b, 'ignored')  # new cached setup,
        self.assertIsNot(ab1, ab3)  # newly computed

        self.cached.cache.enable_caching()  # enable the cached again
        ab4 = self.cached(a, b, 'ignored')  # new cached, ab4 is cached
        self.assertIsNot(ab1, ab4)  # ab1 was deleted a long time ago
        self.assertIsNot(ab3, ab4)  # caching was disabled at ab3
        self.assertIs(ab4, self.cached(a, b, 'its ignored'))  # cached working again?

    def test_reset(self):
        a = ObsAr(np.random.normal(0, 1, (2, 1)))
        b = ObsAr(np.random.normal(0, 1, (2, 1)))
        ab1 = self.cached(a, b, 'ignored')  # cached is ab
        ab2 = self.cached(a, b, 2)  # cached is still ab
        self.assertIs(ab1, ab2)  # thus ab and ab2 are the SAME objects
        self.cached.cache.reset()
        self.assertIsNot(ab1, self.cached(a, b, 2))  # thus ab and ab2 are the SAME objects


class Test(unittest.TestCase):
    def setUp(self):
        def op(x, *args):
            return (x,) + args
        self.cached = Cacher(op, 2)

    def test_pickling(self):
        self.assertRaises(PickleError, self.cached.__getstate__)
        self.assertRaises(PickleError, self.cached.__setstate__)

    def test_copy(self):
        tmp = self.cached.__deepcopy__()
        assert(tmp.operation is self.cached.operation)
        self.assertEqual(tmp.limit, self.cached.limit)

    def test_reset(self):
        opcalls = [0]
        def op(x, y):
            opcalls[0] += 1
            return x + y
        # cacher caches two inputs
        cache = Cacher(op, 1)

        ins = [1, 2]
        b = cache(*ins)
        self.assertIs(cache(*ins), b)
        self.assertEqual(opcalls[0], 1)
        self.assertIn(ins, cache.cached_inputs.values())

        self.assertRaises(TypeError, cache, 'this does not work', 2)

        # And the cacher should reset!
        self.assertDictEqual(cache.cached_input_ids, {},)
        self.assertDictEqual(cache.cached_outputs, {},)
        self.assertDictEqual(cache.inputs_changed, {},)

    def test_cached_atomic_str(self):
        i = "printing the cached value"
        print(self.cached(i))
        self.assertIn((i,), self.cached.cached_outputs.values())
        print(self.cached(i))
        self.assertIn((i,), self.cached.cached_outputs.values())
        self.assertEqual(len(self.cached.cached_outputs.values()), 1)

    def test_caching_non_cachables(self):
        class O(object):
            "not cachable"
        o = O()
        c = self.cached(o, 1)
        # It gives back the right result
        self.assertIs(c[0], o)
        # but does not save the value in the cached
        self.assertEqual(len(self.cached.cached_outputs.values()), 0)

    def test_force_kwargs(self):
        # sum the operands and save a call to operation
        opcalls = [0]
        def op(x, y, force=False):
            opcalls[0] += 1
            return x + y
        # cacher caches two inputs
        cache = Cacher(op, 1, force_kwargs=('force',))

        a = ObsAr(np.random.normal(0, 1, (2, 1)))
        b = ObsAr(np.random.normal(0, 1, (2, 1)))

        ab = cache(a, b)
        self.assertEqual(opcalls[0], 1)
        self.assertIs(ab, cache(a, b))
        self.assertIsNot(ab, cache(a, b, force='given'))
        self.assertEqual(opcalls[0], 2)

    def test_reset_on_operation_error(self):
        # sum the operands and save a call to operation
        opcalls = [0]
        a = ObsAr(np.random.randn())
        b = ObsAr(np.random.randn())
        def op(x, y, force=False):
            opcalls[0] += 1
            print(opcalls)
            if opcalls[0] == 2:
                raise RuntimeError("Oh Noooo, something went wrong here, what is the cacher to do????")
            return x + y
        # cacher caches two inputs
        cache = Cacher(op, 1, force_kwargs=('force',))
        ab = cache(a, b)
        self.assertEqual(ab, a + b)
        self.assertIs(ab, cache(a,b))
        self.assertEqual(opcalls[0], 1)
        self.assertRaises(RuntimeError, cache, b, a)
        self.assertEqual(len(cache.cached_inputs), 0)  # we have reset the cache, bc of error

        # but the caching still works:
        ab = cache(a, b)
        self.assertIs(ab, cache(a, b))

    def test_cached_atomic_int(self):
        i = 1234
        print(self.cached(i))
        self.assertIn((i,), self.cached.cached_outputs.values())
        print(self.cached(i))
        self.assertIn((i,), self.cached.cached_outputs.values())
        self.assertEqual(len(self.cached.cached_outputs.values()), 1)

    def test_cached_ObsAr(self):
        i = ObsAr(np.random.normal(0, 1, (10, 3)))
        # Call cacher:
        self.cached(i)
        _inputs = self.cached.combine_inputs((i,), {}, ())
        id_ = self.cached.prepare_cache_id(_inputs)
        # Test whether the value has been cached:
        self.assertIs(i, self.cached.cached_inputs[id_][0])
        self.assertIs(i, self.cached(i)[0])
        # Change the value:
        i[0] = 10
        # Has it changed?
        self.assertTrue(self.cached.inputs_changed[id_])
        # Call the cacher:
        self.cached(i)
        # Is it now updated?
        self.assertFalse(self.cached.inputs_changed[id_])

    def test_chached_ObsAr_atomic(self):
        i = ObsAr(np.random.normal(0, 1, (10, 3)))
        self.cached(i, 1234)
        # Call cacher:
        self.cached(i, 1234)
        _inputs = self.cached.combine_inputs((i, 1234), {}, ())
        id_ = self.cached.prepare_cache_id(_inputs)
        # Test whether the value has been cached:
        self.assertIs(i, self.cached.cached_inputs[id_][0])
        self.assertIs(i, self.cached(i, 1234)[0])
        self.assertIs(1234, self.cached(i, 1234)[1])
        # Change the value:
        i[0] = 10
        # Has it changed?
        self.assertTrue(self.cached.inputs_changed[id_])
        # Call the cacher:
        old_c = self.cached(i, 1234)
        # Is it now updated?
        self.assertFalse(self.cached.inputs_changed[id_])
        # Put in another number:
        # Stack of cached after next line: 1235, 1234
        old_c1235 = self.cached(i, 1235)
        self.assertIs(old_c1235, self.cached(i, 1235))
        self.assertEqual(len(self.cached.cached_inputs), 2)
        # Put in a str:
        # Stack of cached after next line: 1235, "another"
        another = self.cached(i, "another")
        self.assertIs(self.cached(i, "another"), another)
        self.assertEqual(len(self.cached.cached_inputs), 2)
        # Now 1234 should be off the stack:
        self.assertIsNot(old_c, self.cached(i, 1234))
        self.assertEqual(len(self.cached.cached_inputs), 2)
        # We have not changed the input:
        self.assertFalse(self.cached.inputs_changed[id_])
        i[4] = 3  # Now we have:
        self.assertTrue(self.cached.inputs_changed[id_])
        # We pushed 1235 off the stack, so it should have created a new tuple
        self.assertIsNot(old_c1235, self.cached(i, 1235))
        # This now pushed 'another' off the stack, so it should have created a new tuple
        self.assertIsNot(self.cached(i, "another"), another)

    def test_sum_ObsAr(self):
        # create three random observables
        a = ObsAr(np.random.normal(0, 1, (2, 1)))
        b = ObsAr(np.random.normal(0, 1, (2, 1)))
        c = ObsAr(np.random.normal(0, 1, (2, 1)))

        # sum the operands
        def op(x, y):
            return x + y
        # cacher caches two inputs
        cache = Cacher(op, 2)

        # cacher ids for the three inputs
        _inputs = cache.combine_inputs((a, b), {}, ())
        id_ab = cache.prepare_cache_id(_inputs)
        _inputs = cache.combine_inputs((a, c), {}, ())
        id_ac = cache.prepare_cache_id(_inputs)
        _inputs = cache.combine_inputs((b, c), {}, ())
        id_bc = cache.prepare_cache_id(_inputs)

        # start with testing the change and if the results
        # actually are cached:
        ab = cache(a, b)
        np.testing.assert_array_equal(ab, a + b)
        self.assertIs(cache(a, b), ab)
        self.assertFalse(cache.inputs_changed[id_ab])
        a[:] += np.random.normal(0, 1, (2, 1))
        self.assertTrue(cache.inputs_changed[id_ab])
        abnew = cache(a, b)
        # Redo the id for ab, as the objects have changed (numpy)
        _inputs = cache.combine_inputs((a, b), {}, ())
        id_ab = cache.prepare_cache_id(_inputs)
        self.assertFalse(cache.inputs_changed[id_ab])
        self.assertIsNot(abnew, ab)
        self.assertIs(cache(a, b), abnew)
        np.testing.assert_array_equal(cache(a, b), a + b)

        # Now combine different arrays:
        np.testing.assert_array_equal(cache(a, c), a + c)
        np.testing.assert_array_equal(cache(b, c), b + c)
        # This should have pushed abnew off the stack
        ab = cache(a, b)
        self.assertIsNot(abnew, ab)
        self.assertIs(ab, cache(a, b))
        self.assertIn(id_bc, cache.inputs_changed)
        self.assertIn(id_ab, cache.inputs_changed)
        self.assertNotIn(id_ac, cache.inputs_changed)

    def test_name(self):
        assert(self.cached.__name__ == self.cached.operation.__name__)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
