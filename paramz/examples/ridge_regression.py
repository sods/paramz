'''
Created on 16 Oct 2015

@author: Max Zwiessele
'''
import numpy as np
from ..model import Model
from ..core.observable_array import ObsAr
from ..param import Param
from ..parameterized import Parameterized

class RidgeRegression(Model):
    '''
    Ridge regression with regularization.

    For any regularization to work we to gradient based optimization.
    '''
    def __init__(self, X, Y, regularizer=None, name='ridge_regression'):
        '''
        :param array-like X: the inputs X of the regression problem
        :param array-like Y: the outputs Y
        :param :py:class:`paramz.examples.ridge_regression.Regularizer` regularizer: the regularizer to use
        :param str name: the name of this regression object
        '''
        super(RidgeRegression, self).__init__(name=name)
        self.X = ObsAr(X)
        self.Y = ObsAr(Y)
        self.regularizer = regularizer
        self.link_parameter(self.regularizer)

    @property
    def beta(self):
        return self.regularizer.beta

    def parameters_changed(self):
        self.reg_error = self.Y-self.X.dot(self.beta)
        # gradient for regularizer is already set by the regularizer!
        self.beta.gradient[:] += (-2*self.reg_error*self.X).sum(0)[:,None]
        self._obj = (self.reg_error**2).sum() + self.regularizer.error

    def objective_function(self):
        return self._obj


class Regularizer(Parameterized):
    def __init__(self, lambda_, beta, name='regularizer'):
        super(Regularizer, self).__init__(name=name)
        if not isinstance(beta, Param):
            beta = Param('beta', beta)
        self.beta = beta
        self.lambda_ = lambda_
        self.link_parameter(beta)

    def parameters_changed(self):
        raise NotImplementedError('Set the error `error` and gradient of beta in here')

class Lasso(Regularizer):
    def __init__(self, lambda_, beta, name='Lasso'):
        super(Lasso, self).__init__(lambda_, beta, name)
    def parameters_changed(self):
        self.error = np.sum(np.abs(self.beta))
        self.beta.gradient[:] = np.sign(self.beta)

class Ridge(Regularizer):
    def __init__(self, lambda_, beta, name='Ridge'):
        super(Ridge, self).__init__(lambda_, beta, name)
    def parameters_changed(self):
        self.error = np.sum(self.beta**2)
        self.beta.gradient[:] = 2*self.beta
