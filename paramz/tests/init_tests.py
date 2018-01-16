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
# * Neither the name of paramz.tests.init_tests nor the names of its
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

from paramz import transformations
from paramz.model import Model
from paramz.param import Param
from paramz.parameterized import Parameterized
from paramz.tests.parameterized_tests import P
import numpy as np
import unittest



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
