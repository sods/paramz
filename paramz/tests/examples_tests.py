#===============================================================================
# Copyright (c) 2015, Max Zwiessele
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
# * Neither the name of paramax nor the names of its
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
from ..examples import RidgeRegression, Lasso, Polynomial

class Test2D(unittest.TestCase):

    def testRidgeRegression(self):
        np.random.seed(1000)
        X = np.random.normal(0,1,(20,2))
        beta = np.random.uniform(0,1,(2,1))
        Y = X.dot(beta)
        #Y += np.random.normal(0, .001, Y.shape)
        m = RidgeRegression(X, Y)
        m.regularizer.lambda_ = 0.00001
        self.assertTrue(m.checkgrad())
        m.optimize('scg', gtol=0, ftol=0, xtol=0,max_iters=10)
        m.optimize(max_iters=10)
        np.testing.assert_array_almost_equal(m.regularizer.weights[1], beta[:,0], 4)
        np.testing.assert_array_almost_equal(m.regularizer.weights[0], [0,0], 4)
        np.testing.assert_array_almost_equal(m.gradient, np.zeros(m.weights.size), 4)

    def testLassoRegression(self):
        np.random.seed(12345)
        X = np.random.uniform(0,1,(20,2))
        beta = np.random.normal(0,1,(2,1))
        Y = X.dot(beta)
        #Y += np.random.normal(0, .001, Y.shape)

        m = RidgeRegression(X, Y, regularizer=Lasso(.00001), basis=Polynomial(1))
        self.assertTrue(m.checkgrad())
        m.optimize(max_iters=10)
        np.testing.assert_array_almost_equal(m.regularizer.weights[1], beta[:,0], 3)
        np.testing.assert_array_almost_equal(m.regularizer.weights[0], [0,0], 3)
        np.testing.assert_array_almost_equal(m.gradient, np.zeros(m.weights.size), 3)

#         m = RidgeRegression(X, Y, regularizer=Lasso(.00001, np.ones(X.shape[1])))
#         self.assertTrue(m.checkgrad())
#         m.optimize()
#         np.testing.assert_array_almost_equal(m.regularizer.weights[1], beta, 4)
#         np.testing.assert_array_almost_equal(m.gradient, np.zeros(m.weights.shape[0]), 4)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testRidgeRegression']
    unittest.main()