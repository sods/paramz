'''
Created on Feb 13, 2014

@author: maxzwiessele
'''
import unittest
import numpy as np

from ..core.observable_array import ObsAr
from ..core.index_operations import ParameterIndexOperations
from ..core.nameable import adjust_name_for_printing
from ..core import HierarchyError
from .. import transformations
from ..parameterized import Parameterized
from ..param import Param, ParamConcatenation
from ..model import Model
from unittest.case import SkipTest
from paramz.caching import Cache_this

class ArrayCoreTest(unittest.TestCase):
    def setUp(self):
        self.X = np.random.normal(1,1, size=(100,10))
        self.obsX = ObsAr(self.X)

    def test_init(self):
        X = ObsAr(self.X)
        X2 = ObsAr(X)
        self.assertIs(X, X2, "no new Observable array, when Observable is given")

    def test_slice(self):
        t1 = self.X[2:78]
        t2 = self.obsX[2:78]
        self.assertListEqual(t1.tolist(), t2.tolist(), "Slicing should be the exact same, as in ndarray")

def test_constraints_in_init():
    class Test(Parameterized):
        def __init__(self, name=None, parameters=[], *a, **kw):
            super(Test, self).__init__(name=name)
            self.x = Param('x', np.random.uniform(0,1,(3,4)), transformations.__fixed__)
            self.x[0].constrain_bounded(0,1)
            self.link_parameter(self.x)
            self.x.unfix()
            self.x[1].fix()
    t = Test()
    c = {transformations.Logistic(0,1): np.array([0, 1, 2, 3]), 'fixed': np.array([4, 5, 6, 7])}
    np.testing.assert_equal(t.x.constraints[transformations.Logistic(0,1)], c[transformations.Logistic(0,1)])
    np.testing.assert_equal(t.x.constraints['fixed'], c['fixed'])

def test_parameter_modify_in_init():
    class TestLikelihood(Parameterized):
        def __init__(self, param1 = 2., param2 = 3., param3 = np.random.uniform(size=(2,2,2))):

            super(TestLikelihood, self).__init__("TestLike")
            self.p1 = Param('param1', param1)
            self.p2 = Param('param2', param2)

            self.link_parameter(self.p1)
            self.link_parameter(self.p2)

            self.p1.fix()
            self.p1.unfix()

            self['.*param'].constrain_positive()

            self.p2.constrain_negative()
            self.p1.fix()
            self.p2.constrain_positive()
            self.p2.fix()
            self.p2.constrain_positive()

            self['.*param1'].unconstrain(transformations.Logexp())


    m = TestLikelihood()
    print(m)
    val = m.p1.values.copy()
    assert(m.p1.is_fixed)
    assert(m.constraints[transformations.Logexp()].tolist() == [1])
    m.randomize()
    assert(m.p1 == val)

class P(Parameterized):
    def __init__(self, name, **kwargs):
        super(P, self).__init__(name=name)
        for k, val in kwargs.items():
            self.__setattr__(k, val)
            self.link_parameter(self.__getattribute__(k))

    @Cache_this()
    def heres_johnny(self, ignore=1):
        return 0


class ModelTest(unittest.TestCase):

    def setUp(self):
        class M(Model):
            def __init__(self, name, **kwargs):
                super(M, self).__init__(name=name)
                for k, val in kwargs.items():
                    self.__setattr__(k, val)
                    self.link_parameter(self.__getattribute__(k))
            def objective_function(self):
                return self._obj
            def log_likelihood(self):
                return -self.objective_function()
            def parameters_changed(self):
                self._obj = (self.param_array**2).sum()
                for p in self.parameters:
                    if hasattr(p, 'heres_johnny'):
                        p.heres_johnny()
                self.gradient[:] = 2*self.param_array

        self.testmodel = M('testmodel')
        self.testmodel.kern = P('rbf')
        self.testmodel.likelihood = P('Gaussian_noise', variance=Param('variance', np.random.uniform(0.1, 0.5), transformations.Logexp()))
        self.testmodel.link_parameter(self.testmodel.kern)
        self.testmodel.link_parameter(self.testmodel.likelihood)
        variance=Param('variance', np.random.uniform(0.1, 0.5), transformations.Logexp())
        lengthscale=Param('lengthscale', np.random.uniform(.1, 1, 1), transformations.Logexp())
        self.testmodel.kern.variance = variance
        self.testmodel.kern.lengthscale = lengthscale
        self.testmodel.kern.link_parameter(lengthscale)
        self.testmodel.kern.link_parameter(variance)
        self.testmodel.trigger_update()
        #=============================================================================
        # GP_regression.           |  Value  |  Constraint  |  Prior  |  Tied to
        # rbf.variance             |    1.0  |     +ve      |         |
        # rbf.lengthscale          |    1.0  |     +ve      |         |
        # Gaussian_noise.variance  |    1.0  |     +ve      |         |
        #=============================================================================
    def test_pydot(self):
        try:
            import pydot
            G = self.testmodel.build_pydot()
            testmodel_node_labels = set(['testmodel',
 'lengthscale',
 'variance',
 'Cacher(heres_johnny)\n  limit=5\n  \\#cached=1',
 'rbf',
 'Cacher(heres_johnny)\n  limit=5\n  \\#cached=1',
 'Gaussian_noise',
 'variance'])
            testmodel_edges = set([tuple(e) for e in [['variance', 'Gaussian_noise'],
 ['Gaussian_noise', 'Cacher(heres_johnny)\n  limit=5\n  \\#cached=1'],
 ['rbf', 'rbf'],
 ['Gaussian_noise', 'variance'],
 ['testmodel', 'Gaussian_noise'],
 ['lengthscale', 'rbf'],
 ['rbf', 'lengthscale'],
 ['rbf', 'testmodel'],
 ['variance', 'rbf'],
 ['testmodel', 'rbf'],
 ['testmodel', 'testmodel'],
 ['Gaussian_noise', 'testmodel'],
 ['Gaussian_noise', 'Gaussian_noise'],
 ['rbf', 'variance'],
 ['rbf', 'Cacher(heres_johnny)\n  limit=5\n  \\#cached=1']]])

            self.assertSetEqual(set([n.get_label() for n in G.get_nodes()]), testmodel_node_labels)

            edges = set()
            for e in G.get_edges():
                points = e.obj_dict['points']
                edges.add(tuple(G.get_node(p)[0].get_label() for p in points))

            self.assertSetEqual(edges, testmodel_edges)
        except ImportError:
            raise SkipTest("pydot not available")

    def test_optimize_preferred(self):
        self.testmodel.update_toggle()
        self.testmodel.optimize(messages=True, xtol=0, ftol=0, gtol=1e-6, bfgs_factor=1)
        self.testmodel.optimize(messages=False)
        np.testing.assert_array_less(self.testmodel.gradient, np.ones(self.testmodel.size)*1e-2)
    def test_optimize_scg(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.testmodel.optimize('scg', messages=1, max_f_eval=10, max_iters=100)
            self.testmodel.optimize('scg', messages=0, xtol=0, ftol=0, gtol=1e-6, max_iters=2)
            self.testmodel.optimize('scg', messages=0, xtol=0, ftol=20, gtol=0, max_iters=2)
            self.testmodel.optimize('scg', messages=0, xtol=20, ftol=0, gtol=0, max_iters=2)
        np.testing.assert_array_less(self.testmodel.gradient, np.ones(self.testmodel.size)*1e-1)
    def test_optimize_tnc(self):
        from ..optimization.optimization import opt_tnc
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.testmodel.optimize_restarts(1, messages=1, optimizer=opt_tnc(), verbose=False)
            self.testmodel.optimize('tnc', messages=1, xtol=0, ftol=0, gtol=1e-6)
        np.testing.assert_array_less(self.testmodel.gradient, np.ones(self.testmodel.size)*1e-2)
        self.assertDictEqual(self.testmodel.optimization_runs[-1].__getstate__(), {})
    def test_optimize_rprop(self):
        try:
            import climin
        except ImportError:
            raise SkipTest("climin not installed, skipping test")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.testmodel.optimize('rprop', messages=1)
        np.testing.assert_array_less(self.testmodel.gradient, np.ones(self.testmodel.size)*1e-2)
    def test_optimize_ada(self):
        try:
            import climin
        except ImportError:
            raise SkipTest("climin not installed, skipping test")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.testmodel.trigger_update()
            self.testmodel.optimize('adadelta', messages=1, step_rate=1, momentum=1)
        np.testing.assert_array_less(self.testmodel.gradient, np.ones(self.testmodel.size)*1e-2)
    def test_optimize_org_bfgs(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(divide='ignore'):
                self.testmodel.optimize_restarts(1, messages=0, optimizer='org-bfgs', xtol=0, ftol=0, gtol=1e-6)
                self.testmodel.optimize(messages=1, optimizer='org-bfgs')
        np.testing.assert_array_less(self.testmodel.gradient, np.ones(self.testmodel.size)*1e-2)
    def test_optimize_fix(self):
        self.testmodel.fix()
        self.assertTrue(self.testmodel.checkgrad())
        self.assertTrue(self.testmodel.checkgrad(1))
        self.testmodel.optimize(messages=1)
    def test_optimize_cgd(self):
        self.assertRaises(KeyError, self.testmodel.optimize, 'cgd', messages=1)
    def test_optimize_simplex(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.testmodel.optimize('simplex', messages=1, xtol=0, ftol=0, gtol=1e-6)
            self.testmodel.optimize('simplex', start=self.testmodel.optimizer_array, messages=0)
        np.testing.assert_array_less(self.testmodel.gradient, np.ones(self.testmodel.size)*1e-2)
    def test_optimize_error(self):
        class M(Model):
            def __init__(self, name, **kwargs):
                super(M, self).__init__(name=name)
                for k, val in kwargs.items():
                    self.__setattr__(k, val)
                    self.link_parameter(self.__getattribute__(k))
                self._allowed_failures = 1
            def objective_function(self):
                raise ValueError('Some error occured')
            def log_likelihood(self):
                raise ValueError('Some error occured')
            def parameters_changed(self):
                #self._obj = (self.param_array**2).sum()
                self.gradient[:] = 2*self.param_array
        testmodel = M("test", var=Param('test', np.random.normal(0,1,(20))))
        testmodel.optimize_restarts(2, messages=0, optimizer='org-bfgs', xtol=0, ftol=0, gtol=1e-6, robust=True)
        self.assertRaises(ValueError, testmodel.optimize_restarts, 1, messages=0, optimizer='org-bfgs', xtol=0, ftol=0, gtol=1e-6, robust=False)

    def test_raveled_index(self):
        self.assertListEqual(self.testmodel._raveled_index_for(self.testmodel['.*variance']).tolist(), [1, 2])
        self.assertListEqual(self.testmodel.kern.lengthscale._raveled_index_for(None).tolist(), [0])

    def test_constraints_testmodel(self):
        self.testmodel['.*rbf'].constrain_negative()
        self.assertListEqual(self.testmodel.constraints[transformations.NegativeLogexp()].tolist(), [0,1])

        self.testmodel['.*lengthscale'].constrain_bounded(0,1)
        self.assertListEqual(self.testmodel.constraints[transformations.NegativeLogexp()].tolist(), [1])
        self.assertListEqual(self.testmodel.constraints[transformations.Logistic(0, 1)].tolist(), [0])

        self.testmodel[''].unconstrain_negative()
        self.assertListEqual(self.testmodel.constraints[transformations.NegativeLogexp()].tolist(), [])
        self.assertListEqual(self.testmodel.constraints[transformations.Logistic(0, 1)].tolist(), [0])

        self.testmodel['.*lengthscale'].unconstrain_bounded(0,1)
        self.assertListEqual(self.testmodel.constraints[transformations.Logistic(0, 1)].tolist(), [])

    def test_constraints_set_direct(self):
        self.testmodel['.*rbf'].constrain_negative()
        self.testmodel['.*lengthscale'].constrain_bounded(0,1)
        self.testmodel['.*variance'].fix()
        
        self.assertListEqual(self.testmodel.constraints[transformations.__fixed__].tolist(), [1,2])
        self.assertListEqual(self.testmodel.constraints[transformations.Logistic(0,1)].tolist(), [0])
        self.assertListEqual(self.testmodel.constraints[transformations.NegativeLogexp()].tolist(), [1])
        
        cache_constraints = self.testmodel.constraints.copy()

        self.testmodel.unconstrain()
        self.testmodel.likelihood.fix()

        self.assertListEqual(self.testmodel._fixes_.tolist(), [transformations.UNFIXED, transformations.UNFIXED, transformations.FIXED])

        self.assertListEqual(self.testmodel.constraints[transformations.__fixed__].tolist(), [2])
        self.assertListEqual(self.testmodel.constraints[transformations.Logistic(0,1)].tolist(), [])
        self.assertListEqual(self.testmodel.constraints[transformations.NegativeLogexp()].tolist(), [])

        self.testmodel.constraints = cache_constraints
        self.assertListEqual(self.testmodel.constraints[transformations.__fixed__].tolist(), [1,2])
        self.assertListEqual(self.testmodel.constraints[transformations.Logistic(0,1)].tolist(), [0])
        self.assertListEqual(self.testmodel.constraints[transformations.NegativeLogexp()].tolist(), [1])
        
        self.assertListEqual(self.testmodel._fixes_.tolist(), [transformations.UNFIXED, transformations.FIXED, transformations.FIXED])

        self.assertIs(self.testmodel.constraints, self.testmodel.likelihood.constraints._param_index_ops)
        self.assertIs(self.testmodel.constraints, self.testmodel.kern.constraints._param_index_ops)
        
        #self.assertSequenceEqual(cache_str, str(self.testmodel), None, str)

    def test_updates(self):
        val = float(self.testmodel.objective_function())
        self.testmodel.update_toggle()
        self.testmodel.kern.randomize(np.random.normal, loc=1, scale=.2)
        self.testmodel.likelihood.randomize()
        self.assertEqual(val, self.testmodel.objective_function())
        self.testmodel.update_model(True)
        self.assertNotEqual(val, self.testmodel.objective_function())

    def test_set_gradients(self):
        self.testmodel.gradient = 10.
        np.testing.assert_array_equal(self.testmodel.gradient, 10.)
        self.testmodel.kern.lengthscale.gradient = 15
        np.testing.assert_array_equal(self.testmodel.gradient, [15., 10., 10.])

    def test_fixing_optimize(self):
        self.testmodel.kern.lengthscale.fix()
        val = float(self.testmodel.kern.lengthscale)
        self.testmodel.randomize()
        self.assertEqual(val, self.testmodel.kern.lengthscale)
        self.testmodel.optimize(max_iters=2)

    def test_regular_expression_misc(self):
        self.assertTrue(self.testmodel[''].checkgrad())

        self.testmodel['.*rbf'][:] = 10
        self.testmodel[''][2] = 11

        np.testing.assert_array_equal(self.testmodel.param_array, [10,10,11])
        np.testing.assert_((self.testmodel[''][:2] == [10,10]).all())

        self.testmodel.kern.lengthscale.fix()
        val = float(self.testmodel.kern.lengthscale)
        self.testmodel.randomize()
        self.assertEqual(val, self.testmodel.kern.lengthscale)

        variances = self.testmodel['.*var'].values()
        self.testmodel['.*var'].fix()
        self.testmodel.randomize()
        np.testing.assert_equal(variances, self.testmodel['.*var'].values())

        self.testmodel[''] = 1.0
        self.maxDiff = None

        self.testmodel[''].unconstrain()
        self.assertSequenceEqual(self.testmodel[''].__str__(VT100=False), "  index  |          testmodel.rbf.lengthscale  |  constraints\n  [0]    |                         1.00000000  |             \n  -----  |             testmodel.rbf.variance  |  -----------\n  [0]    |                         1.00000000  |             \n  -----  |  testmodel.Gaussian_noise.variance  |  -----------\n  [0]    |                         1.00000000  |             ")

    def test_fix_unfix(self):
        default_constraints = dict(self.testmodel.constraints.items())
        self.testmodel['.*lengthscale'].fix()
        fixed = self.testmodel.constraints[transformations.__fixed__]
        self.assertListEqual(fixed.tolist(), [0])
        unfixed = self.testmodel.kern.lengthscale.unfix()
        self.testmodel['.*lengthscale'].constrain_positive()
        self.assertListEqual(unfixed.tolist(), [0])

        fixed = self.testmodel['.*rbf'].fix()
        fixed = self.testmodel.constraints[transformations.__fixed__]
        self.assertListEqual(fixed.tolist(), [0,1])

        unfixed = self.testmodel.kern.unfix()
        self.assertListEqual(unfixed.tolist(), [0,1])

        fixed = self.testmodel.constraints[transformations.__fixed__]
        self.testmodel['.*rbf'].unfix()
        np.testing.assert_array_equal(fixed, self.testmodel.constraints[transformations.__fixed__])

        #print default_constraints
        test_constraints = dict(self.testmodel.constraints.items())
        for k in default_constraints:
            np.testing.assert_array_equal(default_constraints[k], test_constraints[k])

    def test_fix_unfix_constraints(self):
        self.testmodel.constrain_bounded(0,1)
        self.testmodel['.*variance'].constrain(transformations.Logexp())
        self.testmodel['.*Gauss'].constrain_bounded(0.3, 0.7)
        before_constraints = dict(self.testmodel.constraints.items())

        self.testmodel.fix()

        test_constraints = dict(self.testmodel.constraints.items())
        for k in before_constraints:
            np.testing.assert_array_equal(before_constraints[k], test_constraints[k])
        np.testing.assert_array_equal(test_constraints[transformations.__fixed__], [0,1,2])


        # Assert fixing works and does not randomize the - say - lengthscale:
        val = float(self.testmodel.kern.lengthscale)
        self.testmodel.randomize()
        self.assertEqual(val, self.testmodel.kern.lengthscale)

        self.testmodel.unfix()

        test_constraints = dict(self.testmodel.constraints.items())
        for k in before_constraints:
            np.testing.assert_array_equal(before_constraints[k], test_constraints[k])

    def test_fix_constrain(self):
        # save the constraints as they where:
        before_constraints = dict(self.testmodel.constraints.items())
        # fix
        self.testmodel.fix()

        test_constraints = dict(self.testmodel.constraints.items())
        # make sure fixes are in place:
        np.testing.assert_array_equal(test_constraints[transformations.__fixed__], [0,1,2])
        # make sure, the constraints still exist
        for k in before_constraints:
            np.testing.assert_array_equal(before_constraints[k], test_constraints[k])

        # override fix and previous constraint:
        self.testmodel.likelihood.constrain_bounded(0,1)
        # lik not fixed anymore
        np.testing.assert_array_equal(self.testmodel.constraints[transformations.__fixed__], [0,1])
        # previous constraints still in place:
        np.testing.assert_array_equal(self.testmodel.constraints[transformations.Logexp()], [0,1])
        # lik bounded
        np.testing.assert_array_equal(self.testmodel.constraints[transformations.Logistic(0,1)], [2])

    def test_caching_offswitch(self):
        self.assertEqual(len(self.testmodel.kern.cache), 1)
        [self.assertEqual(len(c.cached_outputs), 1) for c in self.testmodel.kern.cache.values()]

        self.testmodel.disable_caching()
        self.testmodel.trigger_update()

        [self.assertFalse(c.cacher_enabled) for c in self.testmodel.kern.cache.values()]
        self.assertFalse(self.testmodel.kern.cache.caching_enabled)
        self.assertFalse(self.testmodel.likelihood.cache.caching_enabled)

        self.assertTrue(self.testmodel.checkgrad())

        self.assertEqual(len(self.testmodel.kern.cache), 1)

        [self.assertEqual(len(c.cached_outputs), 0) for c in self.testmodel.kern.cache.values()]

        self.testmodel.enable_caching()
        self.testmodel.trigger_update()

        self.assertEqual(len(self.testmodel.kern.cache), 1)
        [self.assertEqual(len(c.cached_outputs), 1) for c in self.testmodel.kern.cache.values()]


    def test_checkgrad(self):
        self.assertTrue(self.testmodel.checkgrad(1))
        self.assertTrue(self.testmodel.checkgrad())
        self.assertTrue(self.testmodel.rbf.variance.checkgrad(1))
        self.assertTrue(self.testmodel.rbf.variance.checkgrad())
        self.assertTrue(self.testmodel._checkgrad(verbose=1))
        self.assertTrue(self.testmodel._checkgrad(verbose=0))

    def test_printing(self):
        print(self.testmodel.hierarchy_name(False))
        self.assertEqual(self.testmodel.num_params, 2)
        self.assertEqual(self.testmodel.kern.lengthscale.num_params, 0)

    def test_hierarchy_error(self):
        self.assertRaises(HierarchyError, self.testmodel.link_parameter, self.testmodel.parameters[0])
        p2 = P('Gaussian_noise', variance=Param('variance', np.random.uniform(0.1, 0.5), transformations.Logexp()))
        self.testmodel.link_parameter(p2.variance)
        self.assertTrue(self.testmodel.checkgrad())
        self.assertRaises(HierarchyError, self.testmodel.unlink_parameter, p2)
        self.assertRaises(HierarchyError, self.testmodel.unlink_parameter, 'not a parameter')

    def test_set_get(self):
        self.testmodel.likelihood.variance = 10
        self.assertIsInstance(self.testmodel.likelihood.variance, Param)
        np.testing.assert_array_equal(self.testmodel.likelihood[:], [10])

    def test_get_by_name(self):
        self.testmodel.likelihood.variance = 10
        self.assertIsInstance(self.testmodel.likelihood.variance, Param)
        np.testing.assert_array_equal(self.testmodel.likelihood[:], [10])

    def test_likelihood_replicate(self):
        m = self.testmodel
        m2 = self.testmodel.copy(memo={})

        np.testing.assert_array_equal(self.testmodel[:], m2[:])

        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())
        m.randomize()
        m2[:] = m[''].values()
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.randomize()
        m2[''] = m[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.randomize()
        m2[:] = m[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.randomize()
        m2[''] = m['']
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2[:] = m[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

        m.Gaussian_noise.randomize()
        m2[:] = m[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

        m['.*var'] = 2
        m2['.*var'] = m['.*var']
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

        np.testing.assert_array_equal(self.testmodel[''].values(), m2[''].values())
        np.testing.assert_array_equal(self.testmodel[:], m2[''].values())
        np.testing.assert_array_equal(self.testmodel[''].values(), m2[:])
        np.testing.assert_array_equal(self.testmodel['.*variance'].values(), m2['.*variance'].values())
        np.testing.assert_array_equal(self.testmodel['.*len'].values, m2['.*len'].values)
        np.testing.assert_array_equal(self.testmodel['.*rbf'].values(), m2['.*rbf'].values())

    def test_set_empty(self):
        pars = self.testmodel[:].copy()
        self.testmodel.rbf[:] = None
        np.testing.assert_array_equal(self.testmodel[:], pars)

    def test_set_error(self):
        self.assertRaises(ValueError, self.testmodel.__setitem__, slice(None), 'test')

    def test_empty_parameterized(self):
        #print(ParamConcatenation([self.testmodel.rbf, self.testmodel.likelihood.variance]))
        self.testmodel.name = 'anothername'
        self.testmodel.link_parameter(Parameterized('empty'))
        hmm = Parameterized('test')
        self.testmodel.kern.test = hmm
        self.testmodel.kern.link_parameter(hmm)
        self.testmodel.kern.test.link_parameter(Param('test1',1))
        self.assertIsInstance(self.testmodel['.*test1$'], Param)
        self.assertIsInstance(self.testmodel['.*test$'], Parameterized)
        self.assertIsInstance(self.testmodel['.*empty'], Parameterized)
        self.assertIsInstance(self.testmodel['.*test'], ParamConcatenation)
        self.assertIsInstance(self.testmodel['.*rbf$'], Parameterized)
        self.assertIs(self.testmodel['rbf.variance'], self.testmodel.rbf.variance)
        self.assertIs(self.testmodel['rbf$'], self.testmodel.rbf)

    def test_likelihood_set(self):
        m = self.testmodel
        m2 = self.testmodel.copy()
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2.kern.lengthscale = m.kern.lengthscale
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2['.*lengthscale'] = m.kern.lengthscale
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2['.*lengthscale'] = m.kern['.*lengthscale']
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2.kern.lengthscale = m.kern['.*lengthscale']
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        np.testing.assert_array_equal(self.testmodel[''].values(), m2[''].values())
        np.testing.assert_array_equal(self.testmodel[:], m2[''].values())
        np.testing.assert_array_equal(self.testmodel[''].values(), m2[:])
        np.testing.assert_array_equal(self.testmodel['.*variance'].values(), m2['.*variance'].values())
        np.testing.assert_array_equal(self.testmodel['.*len'], m2['.*len'])
        np.testing.assert_array_equal(self.testmodel['.*rbf'][0], m2['.*rbf'][0])
        np.testing.assert_array_equal(self.testmodel['.*rbf'][1], m2['.*rbf'][1])


#class ErrorTest(unittest.TestCase):
#    def test_fail_param_dimension_change(self):
#        p = Param('test', np.random.normal(0,1,2))
#        m = Parameterized('test')
#        self.assertRaises(ValueError, m.link_parameter, p[:,None])


class ParameterizedTest(unittest.TestCase):

    def setUp(self):
        self.rbf = Parameterized('rbf')
        self.rbf.lengthscale = Param('lengthscale', np.random.uniform(.1, 1), transformations.Logexp())
        self.rbf.variance = Param('variance', np.random.uniform(0.1, 0.5), transformations.Logexp())
        self.rbf.link_parameters(self.rbf.variance, self.rbf.lengthscale)

        self.white = P('white', variance=Param('variance', np.random.uniform(0.1, 0.5), transformations.Logexp()))
        self.param = Param('param', np.random.uniform(0,1,(10,5)), transformations.Logistic(0, 1))

        self.test1 = Parameterized('test_parameterized')

        self.test1.param = self.param
        self.test1.kern = Parameterized('add')
        self.test1.kern.link_parameters(self.rbf, self.white)

        self.test1.link_parameter(self.test1.kern)
        self.test1.link_parameter(self.param, 0)

        # print self.test1:
        #=============================================================================
        # test_model.          |    Value    |  Constraint   |  Prior  |  Tied to
        # param                |  (25L, 2L)  |   {0.0,1.0}   |         |
        # add.rbf.variance     |        1.0  |  0.0,1.0 +ve  |         |
        # add.rbf.lengthscale  |        1.0  |  0.0,1.0 +ve  |         |
        # add.white.variance   |        1.0  |  0.0,1.0 +ve  |         |
        #=============================================================================

    def test_original(self):
        self.assertIs(self.test1.param[[0]]._get_original(None), self.param)

    def test_unfixed_param_array(self):
        self.test1.param_array[:] = 0.1
        np.testing.assert_array_equal(self.test1.unfixed_param_array, [0.1]*53)
        self.test1.unconstrain()
        self.test1.kern.rbf.lengthscale.fix()
        np.testing.assert_array_equal(self.test1.kern.unfixed_param_array, [0.1, 0.1])
        np.testing.assert_array_equal(self.test1.unfixed_param_array, [0.1]*52)

    def test_set_param_array(self):
        self.assertRaises(AttributeError, setattr, self.test1, 'param_array', 0)

    def test_fixed_optimizer_copy(self):
        self.test1[:] = 0.1
        self.test1.unconstrain()
        np.testing.assert_array_equal(self.test1.kern.white.optimizer_array, [0.1])
        self.test1.kern.fix()
        #self.assertRaises(ValueError, self.test1.kern.constrain, transformations.__fixed__)

        np.testing.assert_array_equal(self.test1.optimizer_array, [0.1]*50)
        np.testing.assert_array_equal(self.test1.optimizer_array, self.test1.param.optimizer_array)

        self.assertTrue(self.test1.kern.is_fixed)
        self.assertTrue(self.test1.kern.white.is_fixed)
        self.assertTrue(self.test1.kern.white._has_fixes())
        self.assertTrue(self.test1._has_fixes())

        np.testing.assert_array_equal(self.test1.kern.optimizer_array, [])
        np.testing.assert_array_equal(self.test1.kern.white.optimizer_array, [])

    def test_param_names(self):
        self.assertSequenceEqual(self.test1.kern.rbf.parameter_names_flat().tolist(), ['test_parameterized.add.rbf.variance', 'test_parameterized.add.rbf.lengthscale'])

        self.test1.param.fix()
        self.test1.kern.rbf.lengthscale.fix()
        self.assertSequenceEqual(self.test1.parameter_names_flat().tolist(), ['test_parameterized.add.rbf.variance', 'test_parameterized.add.white.variance'])
        self.assertEqual(self.test1.parameter_names_flat(include_fixed=True).size, self.test1.size)

    def test_num_params(self):
        self.assertEqual(self.test1.num_params, 2)
        self.assertEqual(self.test1.add.num_params, 2)
        self.assertEqual(self.test1.add.white.num_params, 1)
        self.assertEqual(self.test1.add.rbf.num_params, 2)

    def test_index_operations(self):
        self.assertRaises(AttributeError, self.test1.add_index_operation, 'constraints', None)
        self.assertRaises(AttributeError, self.test1.remove_index_operation, 'not_an_index_operation')

    def test_names(self):
        self.assertSequenceEqual(self.test1.parameter_names(adjust_for_printing=True), self.test1.parameter_names(adjust_for_printing=False))
        self.test1.unlink_parameter(self.test1.kern)
        newname = 'this@is a+new name!'
        self.test1.kern.name = newname
        self.test1.link_parameter(self.test1.kern)
        self.assertSequenceEqual(self.test1.kern.name, newname)
        self.assertSequenceEqual(self.test1.kern.hierarchy_name(False), 'test_parameterized.'+newname)
        self.assertSequenceEqual(self.test1.kern.hierarchy_name(True), 'test_parameterized.'+adjust_name_for_printing(newname))
        self.assertRaises(NameError, adjust_name_for_printing, '%')

    def test_traverse_parents(self):
        c = []
        self.test1.kern.rbf.traverse_parents(lambda x: c.append(x.name))
        self.assertSequenceEqual(c, ['test_parameterized', 'param', 'add', 'white', 'variance'])
        c = []
        self.test1.kern.white.variance.traverse_parents(lambda x: c.append(x.name))
        self.assertSequenceEqual(c, ['test_parameterized', 'param', 'add', 'rbf', 'variance', 'lengthscale', 'white'])

    def test_names_already_exist(self):
        self.test1.kern.name = 'newname'
        self.test1.p = Param('newname', 1.22345)
        self.test1.link_parameter(self.test1.p)
        self.assertSequenceEqual(self.test1.kern.name, 'newname')
        self.assertSequenceEqual(self.test1.p.name, 'newname_1')
        self.test1.p2 = Param('newname', 1.22345)
        self.test1.link_parameter(self.test1.p2)
        self.assertSequenceEqual(self.test1.p2.name, 'newname_2')
        self.test1.kern.rbf.lengthscale.name = 'variance'
        self.assertSequenceEqual(self.test1.kern.rbf.lengthscale.name, 'variance_1')
        self.test1.kern.rbf.variance.name = 'variance_1'
        self.assertSequenceEqual(self.test1.kern.rbf.lengthscale.name, 'variance_2')
        self.test1.kern.rbf.variance.name = 'variance'
        self.assertSequenceEqual(self.test1.kern.rbf.lengthscale.name, 'variance_2')
        self.assertSequenceEqual(self.test1.kern.rbf.variance.name, 'variance')


    def test_add_parameter(self):
        self.assertEquals(self.rbf._parent_index_, 0)
        self.assertEquals(self.white._parent_index_, 1)
        self.assertEquals(self.param._parent_index_, 0)
        pass

    def test_fixes(self):
        self.white.fix(warning=False)
        self.test1.unlink_parameter(self.param)
        self.assertTrue(self.test1._has_fixes())
        self.assertListEqual(self.test1._fixes_.tolist(),[transformations.UNFIXED,transformations.UNFIXED,transformations.FIXED])
        self.test1.kern.link_parameter(self.white, 0)
        self.assertListEqual(self.test1._fixes_.tolist(),[transformations.FIXED,transformations.UNFIXED,transformations.UNFIXED])
        self.test1.kern.rbf.fix()
        self.assertListEqual(self.test1._fixes_.tolist(),[transformations.FIXED]*3)
        self.test1.fix()
        self.assertTrue(self.test1.is_fixed)
        self.assertListEqual(self.test1._fixes_.tolist(),[transformations.FIXED]*self.test1.size)

    def test_remove_parameter(self):
        self.white.fix()
        self.test1.kern.unlink_parameter(self.white)
        self.assertIs(self.test1._fixes_,None)

        self.assertIsInstance(self.white.constraints, ParameterIndexOperations)
        self.assertListEqual(self.white._fixes_.tolist(), [transformations.FIXED])
        self.assertIs(self.test1.constraints, self.rbf.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.param.constraints._param_index_ops)

        self.test1.link_parameter(self.white, 0)
        self.assertIs(self.test1.constraints, self.white.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.rbf.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.param.constraints._param_index_ops)
        self.assertListEqual(self.test1.constraints[transformations.__fixed__].tolist(), [0])
        self.assertIs(self.white._fixes_,None)
        self.assertListEqual(self.test1._fixes_.tolist(),[transformations.FIXED] + [transformations.UNFIXED] * 52)

        self.test1.unlink_parameter(self.white)
        self.assertIs(self.test1._fixes_,None)
        self.assertListEqual(self.white._fixes_.tolist(), [transformations.FIXED])
        self.assertIs(self.test1.constraints, self.rbf.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.param.constraints._param_index_ops)
        self.assertListEqual(self.test1.constraints[transformations.Logexp()].tolist(), list(range(self.param.size, self.param.size+self.rbf.size)))

    def test_remove_parameter_param_array_grad_array(self):
        val = self.test1.kern.param_array.copy()
        self.test1.kern.unlink_parameter(self.white)
        self.assertListEqual(self.test1.kern.param_array.tolist(), val[:2].tolist())

    def test_add_parameter_already_in_hirarchy(self):
        self.assertRaises(HierarchyError, self.test1.link_parameter, self.white.parameters[0])

    def test_default_constraints(self):
        self.assertIs(self.rbf.variance.constraints._param_index_ops, self.rbf.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.rbf.constraints._param_index_ops)
        self.assertListEqual(self.rbf.constraints.indices()[0].tolist(), list(range(2)))
        kern = self.test1.kern
        self.test1.unlink_parameter(kern)
        self.assertListEqual(kern.constraints[transformations.Logexp()].tolist(), list(range(3)))

    def test_constraints(self):
        self.rbf.constrain(transformations.Square(), False)
        self.assertListEqual(self.test1.constraints[transformations.Square()].tolist(), list(range(self.param.size, self.param.size+self.rbf.size)))
        self.assertListEqual(self.test1.constraints[transformations.Logexp()].tolist(), [self.param.size+self.rbf.size])

        self.test1.kern.unlink_parameter(self.rbf)
        self.assertListEqual(self.test1.constraints[transformations.Square()].tolist(), [])

        self.test1.unconstrain_positive()
        self.assertListEqual(self.test1.constraints[transformations.Logexp()].tolist(), [])

    def test_constraints_link_unlink(self):
        self.test1.unlink_parameter(self.test1.kern)
        self.test1.kern.rbf.unlink_parameter(self.test1.kern.rbf.lengthscale)
        self.test1.kern.rbf.link_parameter(self.test1.kern.rbf.lengthscale)
        self.test1.kern.rbf.unlink_parameter(self.test1.kern.rbf.lengthscale)
        self.test1.link_parameter(self.test1.kern)

    def test_constraints_views(self):
        self.assertEqual(self.white.constraints._offset, self.param.size+self.rbf.size)
        self.assertEqual(self.rbf.constraints._offset, self.param.size)
        self.assertEqual(self.param.constraints._offset, 0)

    def test_fixing_randomize(self):
        self.white.fix(warning=True)
        val = float(self.white.variance)
        self.test1.randomize()
        self.assertEqual(val, self.white.variance)

    def test_randomize(self):
        ps = self.test1.param.view(np.ndarray).copy()
        self.test1.param[2:5].fix()
        self.test1.param.randomize()
        self.assertFalse(np.all(ps==self.test1.param),str(ps)+str(self.test1.param))

    def test_fixing_randomize_parameter_handling(self):
        self.rbf.fix(0.1, warning=True)
        val = float(self.rbf.variance)
        self.test1.kern.randomize()
        self.assertEqual(val, self.rbf.variance)

    def test_add_parameter_in_hierarchy(self):
        self.test1.kern.rbf.link_parameter(Param("NEW", np.random.rand(2), transformations.NegativeLogexp()), 1)
        self.assertListEqual(self.test1.constraints[transformations.NegativeLogexp()].tolist(), list(range(self.param.size+1, self.param.size+1 + 2)))
        self.assertListEqual(self.test1.constraints[transformations.Logistic(0,1)].tolist(), list(range(self.param.size)))
        self.assertListEqual(self.test1.constraints[transformations.Logexp(0,1)].tolist(), np.r_[50, 53:55].tolist())

    def test_checkgrad_hierarchy_error(self):
        self.assertRaises(HierarchyError, self.test1.checkgrad)
        self.assertRaises(HierarchyError, self.test1.kern.white.checkgrad)

    def test_printing(self):
        print(self.test1)
        print(self.param)
        print(self.test1[''])

class InitTests(unittest.TestCase):
    def setUp(self):
        class M(Model):
            def __init__(self, name, **kwargs):
                super(M, self).__init__(name=name)
                for k, val in kwargs.items():
                    self.__setattr__(k, val)
                    self.link_parameter(self.__getattribute__(k))
            def objective_function(self):
                return self._obj
            def log_likelihood(self):
                return -self.objective_function()
            def parameters_changed(self):
                self._obj = (self.param_array**2).sum()
                self.gradient[:] = 2*self.param_array
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.testmodel = M('testmodel', initialize=False)
            self.testmodel.kern = Parameterized('rbf', initialize=False)
            self.testmodel.likelihood = P('Gaussian_noise', variance=Param('variance', np.random.uniform(0.1, 0.5), transformations.Logexp()), initialize=False)
        self.testmodel.link_parameter(self.testmodel.kern)
        self.testmodel.link_parameter(self.testmodel.likelihood)
        variance=Param('variance', np.random.uniform(0.1, 0.5), transformations.Logexp())
        lengthscale=Param('lengthscale', np.random.uniform(.1, 1, 1), transformations.Logexp())
        self.testmodel.kern.variance = variance
        self.testmodel.kern.lengthscale = lengthscale
        self.testmodel.kern.link_parameter(lengthscale)
        self.testmodel.kern.link_parameter(variance)

    def test_initialize(self):
        self.assertFalse(self.testmodel.likelihood._model_initialized_)
        self.assertFalse(self.testmodel.kern._model_initialized_)
        self.assertRaises(AttributeError, self.testmodel.__str__)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # check the warning is being raised
            self.assertRaises(RuntimeWarning, self.testmodel.checkgrad)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # check that the gradient checker just returns false
            self.assertFalse(self.testmodel.checkgrad())
            self.assertFalse(self.testmodel.kern.checkgrad())
        self.testmodel.initialize_parameter()
        self.assertTrue(self.testmodel.likelihood._model_initialized_)
        self.assertTrue(self.testmodel.kern._model_initialized_)
        self.assertTrue(self.testmodel.checkgrad())

    def test_load_initialized(self):
        self.assertFalse(self.testmodel.likelihood._model_initialized_)
        self.assertFalse(self.testmodel.kern._model_initialized_)
        self.assertRaises(AttributeError, self.testmodel.__str__)
        
        # Model is not initialized, so we cannot set parameters:        
        def err():
            self.testmodel[:] = 2.
        self.assertRaises(AttributeError, err)
        
        self.assertFalse(self.testmodel.likelihood._model_initialized_)
        self.assertFalse(self.testmodel.kern._model_initialized_)
        self.assertFalse(self.testmodel._model_initialized_)
        
        import warnings
        #import ipdb;ipdb.set_trace()
        #with warnings.catch_warnings():
        #    warnings.simplefilter("error")
        #    # check the warning is being raised
        #    self.assertRaises(RuntimeWarning, self.testmodel.checkgrad)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # check that the gradient checker just returns false
            self.assertFalse(self.testmodel.checkgrad())
            self.assertFalse(self.testmodel.kern.checkgrad())
        
        # Set updates off, so we do not call the expensive algebra
        self.testmodel.update_model(False)
        
        # Still not initialized the model, so setting should not work:
        self.assertRaises(AttributeError, err)
        
        # Now initialize the parameter connections:
        self.testmodel.initialize_parameter()
        # And set parameters, without updating
        self.assertIsNone(err())

        # Model has not been updated, even once, but the parameters are connected, so it tries to
        # access the log likelihood, which does not exist:
        self.assertRaises(AttributeError, self.testmodel.checkgrad)
        self.assertRaises(AttributeError, self.testmodel.kern.checkgrad)
        
        # parameters are initialized
        self.assertTrue(self.testmodel.likelihood._model_initialized_)
        self.assertTrue(self.testmodel.kern._model_initialized_)
        
        # update the model now and check everything is working as expected:
        
        self.testmodel.update_model(True)
        
        np.testing.assert_allclose(self.testmodel.param_array, 2., 1e-4) 
        self.assertTrue(self.testmodel.checkgrad())
         

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_add_parameter']
    unittest.main()
