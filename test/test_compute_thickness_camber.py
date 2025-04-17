"""
Unit tests for the compute_thickness_camber function.

Uses pytest for testing.
"""

import numpy as np
import pytest

from DASC500.formulas.airfoil.compute_thickness_camber import compute_thickness_camber  # Replace your_module

def test_compute_thickness_camber_simple():
    """Test with a simple airfoil shape."""
    points = np.array([
        [0, 0], [1, 1], [2, 0],  # Upper surface
        [0, 0], [1, -1], [2, 0]  # Lower surface
    ])
    x_coords, thickness, camber = compute_thickness_camber(points)
    expected_x = np.array([0, 1, 2])
    expected_thickness = np.array([0, 2, 0])
    expected_camber = np.array([0, 0, 0])

    assert np.array_equal(x_coords, expected_x)
    assert np.array_equal(thickness, expected_thickness)
    assert np.array_equal(camber, expected_camber)

def test_compute_thickness_camber_non_uniform_x():
    """Test with non-uniformly spaced x-coordinates."""
    points = np.array([
        [0, 0], [1, 1], [2, 0], [1.5, 0.5], #Upper surface
        [0, 0], [1, -1], [2, 0], [1.5, -0.5] #Lower surface
    ])
    x_coords, thickness, camber = compute_thickness_camber(points)
    expected_x = np.array([0, 1, 1.5, 2])
    expected_thickness = np.array([0, 2, 1, 0])
    expected_camber = np.array([0, 0, 0, 0])

    assert np.array_equal(x_coords, expected_x)
    assert np.array_equal(thickness, expected_thickness)
    assert np.array_equal(camber, expected_camber)

def test_compute_thickness_camber_cambered():
    """Test with a cambered airfoil shape."""
    points = np.array([
        [0, 0.1], [1, 1.1], [2, 0.1], #Upper surface
        [0, -0.1], [1, -0.9], [2, -0.1] #Lower surface
    ])
    x_coords, thickness, camber = compute_thickness_camber(points)
    expected_x = np.array([0, 1, 2])
    expected_thickness = np.array([0.2, 2, 0.2])
    expected_camber = np.array([0, 0.1, 0])

    assert np.array_equal(x_coords, expected_x)
    assert np.array_equal(thickness, expected_thickness)
    assert np.allclose(camber, expected_camber, rtol=1e-1)

def test_compute_thickness_camber_single_x():
    """Test with only one x-coordinate."""
    points = np.array([
        [0, 1], [0, -1]
    ])
    x_coords, thickness, camber = compute_thickness_camber(points)
    expected_x = np.array([0])
    expected_thickness = np.array([2])
    expected_camber = np.array([0])

    assert np.array_equal(x_coords, expected_x)
    assert np.array_equal(thickness, expected_thickness)
    assert np.array_equal(camber, expected_camber)

def test_compute_thickness_camber_unsorted_points():
    """Test with unsorted input points."""
    points = np.array([
        [1, 1], [0, 0], [2, 0], [1, -1], [2, 0], [0, 0]
    ])
    x_coords, thickness, camber = compute_thickness_camber(points)
    expected_x = np.array([0, 1, 2])
    expected_thickness = np.array([0, 2, 0])
    expected_camber = np.array([0, 0, 0])

    assert np.array_equal(x_coords, expected_x)
    assert np.array_equal(thickness, expected_thickness)
    assert np.array_equal(camber, expected_camber)

def test_compute_thickness_camber_duplicate_y():
    """Test with duplicate y-values for the same x."""
    points = np.array([
        [0, 0], [1, 1], [1, 1], [2, 0],
        [0, 0], [1, -1], [1, -1], [2, 0]
    ])
    x_coords, thickness, camber = compute_thickness_camber(points)
    expected_x = np.array([0, 1, 2])
    expected_thickness = np.array([0, 2, 0])
    expected_camber = np.array([0, 0, 0])

    assert np.array_equal(x_coords, expected_x)
    assert np.array_equal(thickness, expected_thickness)
    assert np.array_equal(camber, expected_camber)