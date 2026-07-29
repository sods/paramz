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


def test_parent_gradient_storage_is_visible_from_parameter_leaves():
    root, child = make_hierarchy()

    root.gradient[:] = [11.0, 12.0, 13.0]

    np.testing.assert_array_equal(child.left.gradient, [11.0, 12.0])
    np.testing.assert_array_equal(child.right.gradient, [13.0])


def test_concatenated_parameter_assignment_synchronizes_root():
    root, child = make_hierarchy()

    root['child\\.(left|right)'][:] = [14.0, 15.0, 16.0]

    np.testing.assert_array_equal(root.param_array, [14.0, 15.0, 16.0])
    np.testing.assert_array_equal(child.left, [14.0, 15.0])
    np.testing.assert_array_equal(child.right, [16.0])


def test_observable_array_scalar_ufunc_returns_a_numpy_scalar():
    from ..core.observable_array import ObsAr

    result = np.add(ObsAr([1.0]), 2.0)[0]

    assert isinstance(result, np.floating)
    assert result == 3.0
