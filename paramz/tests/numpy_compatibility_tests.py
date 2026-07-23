import numpy as np

from ..param import Param
from ..parameterized import Parameterized


def make_hierarchy():
    child = Parameterized('child', [Param('left', [1.0, 2.0]), Param('right', [3.0])])
    return Parameterized('root', [child]), child


def test_root_assignment_synchronizes_parameter_leaves():
    root, child = make_hierarchy()

    root[:] = [10.0, 20.0, 30.0]

    np.testing.assert_array_equal(child.left, [10.0, 20.0])
    np.testing.assert_array_equal(child.right, [30.0])


def test_leaf_assignment_synchronizes_flat_parent_arrays():
    root, child = make_hierarchy()

    child.left[:] = [4.0, 5.0]

    np.testing.assert_array_equal(child.param_array, [4.0, 5.0, 3.0])
    np.testing.assert_array_equal(root.param_array, [4.0, 5.0, 3.0])


def test_gradient_storage_remains_shared_across_hierarchy():
    root, child = make_hierarchy()

    child.left.gradient = [7.0, 8.0]

    np.testing.assert_array_equal(child.gradient[:2], [7.0, 8.0])
    np.testing.assert_array_equal(root.gradient[:2], [7.0, 8.0])
