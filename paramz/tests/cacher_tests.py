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

class Test(unittest.TestCase):
    def setUp(self):
        def op(x, *args):
            return (x,)+args
        self.cache = Cacher(op, 2)

    def test_pickling(self):
        self.assertRaises(PickleError, self.cache.__getstate__)
        self.assertRaises(PickleError, self.cache.__setstate__)

    def test_copy(self):
        tmp = self.cache.__deepcopy__()
        assert(tmp.operation is self.cache.operation)
        self.assertEqual(tmp.limit, self.cache.limit)

    def test_reset(self):
        opcalls = [0]
        def op(x, y):
            opcalls[0] += 1
            return x+y
        # cacher caches two inputs
        cache = Cacher(op, 1)

        ins = [1,2]
        b = cache(*ins)
        self.assertIs(cache(*ins), b)
        self.assertEqual(opcalls[0], 1)
        self.assertIn(ins, cache.cached_inputs.values())
        
        self.assertRaises(TypeError, cache, 'this does not work', 2)

        # And the cacher should reset!
        self.assertDictEqual(cache.cached_input_ids, {}, )
        self.assertDictEqual(cache.cached_outputs, {}, )
        self.assertDictEqual(cache.inputs_changed, {}, )
        
    def test_cached_atomic_str(self):
        i = "printing the cached value"
        print(self.cache(i))
        self.assertIn((i,), self.cache.cached_outputs.values())
        print(self.cache(i))
        self.assertIn((i,), self.cache.cached_outputs.values())
        self.assertEqual(len(self.cache.cached_outputs.values()), 1)

    def test_caching_non_cachables(self):
        class O(object):
            "not cachable"
        o = O()
        c = self.cache(o, 1)
        # It gives back the right result
        self.assertIs(c[0], o)
        # but does not save the value in the cache
        self.assertEqual(len(self.cache.cached_outputs.values()), 0)

    def test_decorator(self):
        opcalls = [0, 0, 0]
        class O(object):
            @Cache_this(ignore_args=[0,3], force_kwargs=('force',))
            def __call__(self, x, y, ignore_this, force=False):
                opcalls[0] += 1
                opcalls[1] = ignore_this
                if force is not False:
                    opcalls[2] = force
                return x+y
        cache = O()

        a = ObsAr(np.random.normal(0,1,(2,1)))
        b = ObsAr(np.random.normal(0,1,(2,1)))

        ab = cache(a,b,'ignored')
        self.assertEqual(opcalls[0], 1)
        self.assertEqual(opcalls[1], 'ignored')
        self.assertIs(ab, cache(a,b,2))
        self.assertEqual(opcalls[1], 'ignored')
        self.assertEqual(opcalls[0], 1)
        abnew = cache(a,b,3,force='given')
        self.assertEqual(opcalls[0], 2)
        self.assertEqual(opcalls[1], 3)
        self.assertEqual(opcalls[2], 'given')
        self.assertIsNot(ab, abnew)
        np.testing.assert_array_equal(abnew, ab)
        abforced = cache(a,b,4,force='given2')
        self.assertEqual(opcalls[0], 3)
        self.assertEqual(opcalls[1], 4)
        self.assertEqual(opcalls[2], 'given2')
        np.testing.assert_array_equal(abnew, abforced)
    
    
    def test_force_kwargs(self):
        # sum the operands and save a call to operation
        opcalls = [0]
        def op(x, y, force=False):
            opcalls[0] += 1
            return x+y
        # cacher caches two inputs
        cache = Cacher(op, 1, force_kwargs=('force',))

        a = ObsAr(np.random.normal(0,1,(2,1)))
        b = ObsAr(np.random.normal(0,1,(2,1)))

        ab = cache(a,b)
        self.assertEqual(opcalls[0], 1)
        self.assertIs(ab, cache(a,b))
        self.assertIsNot(ab, cache(a,b,force='given'))
        self.assertEqual(opcalls[0], 2)
        
    def test_reset_on_operation_error(self):
        # sum the operands and save a call to operation
        opcalls = [0]
        
        def op(x, y, force=False):
            opcalls[0] += 1
            return x+y
        # cacher caches two inputs
        cache = Cacher(op, 1, force_kwargs=('force',))


    def test_cached_atomic_int(self):
        i = 1234
        print(self.cache(i))
        self.assertIn((i,), self.cache.cached_outputs.values())
        print(self.cache(i))
        self.assertIn((i,), self.cache.cached_outputs.values())
        self.assertEqual(len(self.cache.cached_outputs.values()), 1)

    def test_cached_ObsAr(self):
        i = ObsAr(np.random.normal(0,1,(10,3)))
        # Call cacher:
        self.cache(i)
        _inputs = self.cache.combine_inputs((i,), {}, ())
        id_ = self.cache.prepare_cache_id(_inputs)
        # Test whether the value has been cached:
        self.assertIs(i, self.cache.cached_inputs[id_][0])
        self.assertIs(i, self.cache(i)[0])
        # Change the value:
        i[0] = 10
        # Has it changed?
        self.assertTrue(self.cache.inputs_changed[id_])        
        # Call the cacher:
        self.cache(i)
        # Is it now updated?
        self.assertFalse(self.cache.inputs_changed[id_])

    def test_chached_ObsAr_atomic(self):
        i = ObsAr(np.random.normal(0,1,(10,3)))
        self.cache(i, 1234)
        # Call cacher:
        self.cache(i, 1234)
        _inputs = self.cache.combine_inputs((i, 1234), {}, ())
        id_ = self.cache.prepare_cache_id(_inputs)
        # Test whether the value has been cached:
        self.assertIs(i, self.cache.cached_inputs[id_][0])
        self.assertIs(i, self.cache(i, 1234)[0])
        self.assertIs(1234, self.cache(i, 1234)[1])
        # Change the value:
        i[0] = 10
        # Has it changed?
        self.assertTrue(self.cache.inputs_changed[id_])        
        # Call the cacher:
        old_c = self.cache(i, 1234)
        # Is it now updated?
        self.assertFalse(self.cache.inputs_changed[id_])
        # Put in another number:
        # Stack of cache after next line: 1235, 1234
        old_c1235 = self.cache(i, 1235)
        self.assertIs(old_c1235, self.cache(i, 1235))
        self.assertEqual(len(self.cache.cached_inputs), 2)
        # Put in a str:
        # Stack of cache after next line: 1235, "another"        
        another = self.cache(i, "another")
        self.assertIs(self.cache(i, "another"), another)
        self.assertEqual(len(self.cache.cached_inputs), 2)
        # Now 1234 should be off the stack:
        self.assertIsNot(old_c, self.cache(i, 1234))
        self.assertEqual(len(self.cache.cached_inputs), 2)
        # We have not changed the input:
        self.assertFalse(self.cache.inputs_changed[id_])
        i[4] = 3 # Now we have:
        self.assertTrue(self.cache.inputs_changed[id_])
        # We pushed 1235 off the stack, so it should have created a new tuple
        self.assertIsNot(old_c1235, self.cache(i, 1235))
        # This now pushed 'another' off the stack, so it should have created a new tuple
        self.assertIsNot(self.cache(i, "another"), another)
        
    def test_sum_ObsAr(self):
        # create three random observables
        a = ObsAr(np.random.normal(0,1,(2,1)))
        b = ObsAr(np.random.normal(0,1,(2,1)))
        c = ObsAr(np.random.normal(0,1,(2,1)))
        
        # sum the operands
        def op(x, y):
            return x+y
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
        ab = cache(a,b)
        np.testing.assert_array_equal(ab, a+b)
        self.assertIs(cache(a, b), ab)
        self.assertFalse(cache.inputs_changed[id_ab])
        a[:] += np.random.normal(0,1,(2,1))
        self.assertTrue(cache.inputs_changed[id_ab])
        abnew = cache(a, b)
        # Redo the id for ab, as the objects have changed (numpy)
        _inputs = cache.combine_inputs((a, b), {}, ())
        id_ab = cache.prepare_cache_id(_inputs)
        self.assertFalse(cache.inputs_changed[id_ab])
        self.assertIsNot(abnew, ab)
        self.assertIs(cache(a, b), abnew)
        np.testing.assert_array_equal(cache(a, b), a+b)

        # Now combine different arrays:
        np.testing.assert_array_equal(cache(a,c), a+c)
        np.testing.assert_array_equal(cache(b,c), b+c)
        # This should have pushed abnew off the stack
        ab = cache(a, b)
        self.assertIsNot(abnew, ab)
        self.assertIs(ab, cache(a, b))
        self.assertIn(id_bc, cache.inputs_changed)
        self.assertIn(id_ab, cache.inputs_changed)
        self.assertNotIn(id_ac, cache.inputs_changed)        
        
    def test_name(self):
        assert(self.cache.__name__ == self.cache.operation.__name__)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()