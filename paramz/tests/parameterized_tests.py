'''
Created on Feb 13, 2014

@author: maxzwiessele
'''
import unittest
import numpy as np

from paramz.core.index_operations import ParameterIndexOperations
from paramz.core.nameable import adjust_name_for_printing
from paramz.core import HierarchyError
from paramz import transformations
from paramz.parameterized import Parameterized
from paramz.param import Param
from paramz.model import Model
from paramz.caching import Cache_this


class P(Parameterized):
    def __init__(self, name, **kwargs):
        super(P, self).__init__(name=name)
        for k, val in kwargs.items():
            self.__setattr__(k, val)
            self.link_parameter(self.__getattribute__(k))

    @Cache_this()
    def heres_johnny(self, ignore=1):
        return 0


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
        self.assertRaisesRegexp(AttributeError, "An index operation with the name constraints was already taken", self.test1.add_index_operation, 'constraints', None)
        self.assertRaisesRegexp(AttributeError, "No index operation with the name", self.test1.remove_index_operation, 'not_an_index_operation')

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

    def test_recursion_limit(self):
        # Recursion limit reached for unnamed kernels:
        def max_recursion():
            kerns = [P('rbf', lengthscale=Param('lengthscale', 1), variance=Param('variance', 1)) for i in range(20)]
            p = Parameterized('add')
            p.link_parameters(*kerns)
        import sys
        sys.setrecursionlimit(100)
        try:
            from builtins import RecursionError as RE
        except:
            RE = RuntimeError
        self.assertRaisesRegexp(RE, "aximum recursion depth", max_recursion)
        # Recursion limit not reached if kernels are named individually:
        sys.setrecursionlimit(1000)
        p = Parameterized('add')
        kerns = [P('rbf_{}'.format(i), lengthscale=Param('lengthscale', 1), variance=Param('variance', 1)) for i in range(10)]
        p.link_parameters(*kerns)

    def test_add_parameter(self):
        self.assertEqual(self.rbf._parent_index_, 0)
        self.assertEqual(self.white._parent_index_, 1)
        self.assertEqual(self.param._parent_index_, 0)
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
        self.assertRaisesRegexp(HierarchyError, "You cannot add a parameter twice into the hierarchy", self.test1.link_parameter, self.white.parameters[0])

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



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_add_parameter']
    unittest.main()
