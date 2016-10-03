'''
Created on 13 Mar 2014

@author: maxz
'''
import unittest, pickle, tempfile, os, paramz
import numpy as np
from ..core.index_operations import ParameterIndexOperations, ParameterIndexOperationsView
from ..core.observable_array import ObsAr
from paramz.transformations import Exponent, Logexp
from ..parameterized import Parameterized
from ..param import Param

class ListDictTestCase(unittest.TestCase):
    def assertListDictEquals(self, d1, d2, msg=None):
        #py3 fix
        #for k,v in d1.iteritems():
        for k,v in d1.items():
            self.assertListEqual(list(v), list(d2[k]), msg)
    def assertArrayListEquals(self, l1, l2):
        for a1, a2 in zip(l1,l2):
            np.testing.assert_array_equal(a1, a2)

class Test(ListDictTestCase):
    def test_parameter_index_operations(self):
        pio = ParameterIndexOperations(dict(test1=np.array([4,3,1,6,4]), test2=np.r_[2:130]))
        piov = ParameterIndexOperationsView(pio, 20, 250)
        #py3 fix
        #self.assertListDictEquals(dict(piov.items()), dict(piov.copy().iteritems()))
        self.assertListDictEquals(dict(piov.items()), dict(piov.copy().items()))

        #py3 fix
        #self.assertListDictEquals(dict(pio.iteritems()), dict(pio.copy().items()))
        self.assertListDictEquals(dict(pio.items()), dict(pio.copy().items()))

        self.assertArrayListEquals(pio.copy().indices(), pio.indices())
        self.assertArrayListEquals(piov.copy().indices(), piov.indices())

        with tempfile.TemporaryFile('w+b') as f:
            pickle.dump(pio, f)
            f.seek(0)
            pio2 = pickle.load(f)
            self.assertListDictEquals(pio._properties, pio2._properties)

        with tempfile.TemporaryFile('w+b') as f:
            pickle.dump(piov, f)
            f.seek(0)
            pio2 = paramz.load(f)
            #py3 fix
            #self.assertListDictEquals(dict(piov.items()), dict(pio2.iteritems()))
            self.assertListDictEquals(dict(piov.items()), dict(pio2.items()))

    def test_param(self):
        param = Param('test', np.arange(4*2).reshape(4,2))
        param[0].constrain_positive()
        param[1].fix()
        pcopy = param.copy()
        self.assertListEqual(param.tolist(), pcopy.tolist())
        self.assertListEqual(str(param).split('\n'), str(pcopy).split('\n'))
        self.assertIsNot(param, pcopy)
        with tempfile.TemporaryFile('w+b') as f:
            pickle.dump(param, f)
            f.seek(0)
            pcopy = paramz.load(f)
        self.assertListEqual(param.tolist(), pcopy.tolist())
        self.assertSequenceEqual(str(param), str(pcopy))

    def test_observable_array(self):
        obs = ObsAr(np.arange(4*2).reshape(4,2))
        pcopy = obs.copy()
        self.assertListEqual(obs.tolist(), pcopy.tolist())
        tmpfile = ''.join(map(str, np.random.randint(10, size=20)))
        try:
            obs.pickle(tmpfile)
            pcopy = paramz.load(tmpfile)
        except:
            raise
        finally:
            os.remove(tmpfile)
        self.assertListEqual(obs.tolist(), pcopy.tolist())
        self.assertSequenceEqual(str(obs), str(pcopy))

    def test_parameterized(self):
        par = Parameterized('parameterized')
        p2 = Parameterized('rbf')
        p2.p1 = Param('lengthscale', np.random.uniform(0.1,.5,3), Exponent())
        p2.link_parameter(p2.p1)
        par.p1 = p2
        par.p2 = Param('linear', np.random.uniform(0.1, .5, 2), Logexp())
        par.link_parameters(par.p1, par.p2)

        par.gradient = 10
        par.randomize()
        pcopy = par.copy()
        self.assertIsInstance(pcopy.constraints, ParameterIndexOperations)
        self.assertIsInstance(pcopy.rbf.constraints, ParameterIndexOperationsView)
        self.assertIs(pcopy.constraints, pcopy.rbf.constraints._param_index_ops)
        self.assertIs(pcopy.constraints, pcopy.rbf.lengthscale.constraints._param_index_ops)
        self.assertListEqual(par.param_array.tolist(), pcopy.param_array.tolist())
        pcopy.gradient = 10 # gradient does not get copied anymore
        self.assertListEqual(par.gradient_full.tolist(), pcopy.gradient_full.tolist())
        self.assertSequenceEqual(str(par), str(pcopy))
        self.assertIsNot(par.param_array, pcopy.param_array)
        self.assertIsNot(par.gradient_full, pcopy.gradient_full)
        with tempfile.TemporaryFile('w+b') as f:
            par.pickle(f)
            f.seek(0)
            pcopy = paramz.load(f)
        self.assertListEqual(par.param_array.tolist(), pcopy.param_array.tolist())
        pcopy.gradient = 10
        np.testing.assert_allclose(par.linear.gradient_full, pcopy.linear.gradient_full)
        np.testing.assert_allclose(pcopy.linear.gradient_full, 10)
        self.assertSequenceEqual(str(par), str(pcopy))


    def _callback(self, what, which):
        what.count += 1


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_parameter_index_operations']
    unittest.main()
