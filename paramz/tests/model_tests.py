#===============================================================================
# Copyright (c) 2018, Max Zwiessele
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
# * Neither the name of paramz.tests.model_tests nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
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
import unittest
import numpy as np

from paramz.core import HierarchyError
from paramz import transformations
from paramz.parameterized import Parameterized
from paramz.param import Param, ParamConcatenation
from paramz.model import Model
from unittest.case import SkipTest
from paramz.tests.parameterized_tests import P, M

class ModelTest(unittest.TestCase):

    def setUp(self):

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
        from paramz.optimization.optimization import opt_tnc
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.testmodel.optimize_restarts(1, messages=1, optimizer=opt_tnc(), verbose=False)
            self.testmodel.optimize('tnc', messages=1, xtol=0, ftol=0, gtol=1e-6)
        np.testing.assert_array_less(self.testmodel.gradient, np.ones(self.testmodel.size)*1e-2)
        # self.assertDictEqual(self.testmodel.optimization_runs[-1].__getstate__(), {})

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

    def test_optimize_adam(self):
        try:
            import climin
        except ImportError:
            raise SkipTest("climin not installed, skipping test")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.testmodel.trigger_update()
            self.testmodel.optimize('adam', messages=1, step_rate=1., momentum=1.)
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

    def test_optimize_restarts(self):
        m = self.testmodel.copy()
        m.optimize_restarts(2, messages=0, xtol=0, ftol=0, gtol=1e-6, robust=False)
        np.testing.assert_array_less(m.gradient, np.ones(self.testmodel.size)*1e-2)
        self.assertIs(len(m.optimization_runs), 2)

    def test_optimize_restarts_parallel(self):
        m = self.testmodel.copy()
        m.optimize_restarts(2, messages=0, xtol=0, ftol=0, gtol=1e-6, robust=False, parallel=True)
        np.testing.assert_array_less(m.gradient, np.ones(self.testmodel.size) * 1e-2)
        self.assertIs(len(m.optimization_runs), 2)

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
