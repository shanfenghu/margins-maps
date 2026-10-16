import numpy as np

from margins_maps.workflow import incidence, incidence_apply, solve_bounds


def test_incidence_matches_direct_application():
    edges = [(0, 0), (0, 1), (1, 0), (1, 1)]
    values = np.array([1.0, 2.0, 3.0, 4.0])
    matrix = incidence(2, 2, edges)
    assert np.allclose(matrix @ values, incidence_apply(2, 2, edges, values))


def test_bounds_recover_two_by_two_extremes():
    edges = [(0, 0), (0, 1), (1, 0), (1, 1)]
    matrix = incidence(2, 2, edges)
    margins = np.array([3.0, 3.0, 4.0, 2.0])
    selector = np.array([1.0, 0.0, 0.0, 0.0])
    lower, upper, low_world, high_world = solve_bounds(matrix, margins, selector)
    assert (lower, upper) == (1.0, 3.0)
    assert np.allclose(matrix @ low_world, margins)
    assert np.allclose(matrix @ high_world, margins)


def test_infeasible_support_returns_none():
    edges = [(0, 0), (1, 1)]
    matrix = incidence(2, 2, edges)
    margins = np.array([1.0, 1.0, 2.0, 0.0])
    selector = np.array([1.0, 0.0])
    assert solve_bounds(matrix, margins, selector) is None
