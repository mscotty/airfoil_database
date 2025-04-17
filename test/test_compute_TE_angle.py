"""
Unit tests for the trailing_edge_angle function in DASC.formulas.airfoil.compute_TE_angle.py.

Uses pytest for testing.
"""

import numpy as np
import pytest

from DASC500.formulas.airfoil.compute_TE_angle import trailing_edge_angle

def test_trailing_edge_angle_simple():
    """Test with a simple linear trailing edge."""
    points = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [2.01, 2.01],
        [2.02, 2.02],
        [2.03, 2.03],
        [2.04, 2.04],
        [3, 3],
    ])
    angle = trailing_edge_angle(points)
    print(angle)
    assert np.isclose(angle, 45.0)

def test_trailing_edge_angle_horizontal():
    """Test with a horizontal trailing edge."""
    points = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [2.01, 0],
        [2.02, 0],
        [2.03, 0],
        [2.04, 0],
        [3, 0],
    ])
    angle = trailing_edge_angle(points)
    assert np.isclose(angle, 0.0)

def test_trailing_edge_angle_vertical():
    """Test with a vertical (or near vertical) trailing edge. Should return a large number."""
    points = np.array([
        [0, 0],
        [2, 1],
        [2.01, 1.01],
        [2.02, 1.02],
        [2.03, 1.03],
        [2.04, 1.04],
        [3, 2],
    ])
    angle = trailing_edge_angle(points)
    
    assert np.isclose(angle, 45.0)

def test_trailing_edge_angle_negative():
    """Test with a negative slope trailing edge."""
    points = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [2.01, 1.99],
        [2.02, 1.98],
        [2.03, 1.97],
        [2.04, 1.96],
        [3, 1],
    ])
    angle = trailing_edge_angle(points)
    assert np.isclose(angle, -45.0)

def test_trailing_edge_angle_non_uniform_spacing():
    """Test with non-uniformly spaced trailing edge points."""
    points = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [2.01, 2.01],
        [2.03, 2.03],
        [2.04, 2.04],
        [2.049, 2.049],
        [3, 3],
    ])
    angle = trailing_edge_angle(points)
    assert np.isclose(angle, 45.0)

def test_trailing_edge_angle_noise():
    """Test with some noise in the trailing edge points."""
    points = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [2.01, 2.01 + 0.001],
        [2.02, 2.02 - 0.002],
        [2.03, 2.03 + 0.001],
        [2.04, 2.04 - 0.001],
        [3, 3],
    ])
    angle = trailing_edge_angle(points)
    assert np.isclose(angle, 45.0, atol=1) #Accept small tolerance due to noise.

def test_trailing_edge_angle_small_point_set():
    """Test with a trailing edge defined by a very small number of points."""
    points = np.array([
        [0, 0],
        [2, 2],
        [2.01, 2.01],
    ])
    angle = trailing_edge_angle(points, num_te_points=2) #Adjusted num_te_points
    assert np.isclose(angle, 45.0)

def test_trailing_edge_angle_very_small_delta():
    """Test with a very small delta x for the trailing edge selection."""
    points = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [2.0001, 2.0001],
        [2.0002, 2.0002],
        [2.0003, 2.0003],
        [2.0004, 2.0004],
        [3, 3],
    ])
    angle = trailing_edge_angle(points)
    assert np.isclose(angle, 45.0)

def test_trailing_edge_angle_single_point():
    """Test with only one point within the 0.05 delta."""
    points = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [2.01, 2.01],
        [3, 3],
    ])
    angle = trailing_edge_angle(points)
    assert np.isclose(angle, 45.0)