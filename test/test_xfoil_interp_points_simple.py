"""
Unit tests for the interpolate_points function.

Uses pytest for testing.
"""

import numpy as np
import pytest
from scipy.interpolate import splprep, splev

from DASC500.xfoil.interpolate_points import interpolate_points  # Replace your_module

def test_interpolate_points_simple():
    """Test with a simple linear curve."""
    points = np.array([[0, 0], [1, 1], [2, 2]])
    interpolated_points = interpolate_points(points, num_points=5)
    assert len(interpolated_points) == 5
    assert np.allclose(interpolated_points[:, 0], np.linspace(0, 2, 5))

def test_interpolate_points_curve():
    """Test with a curved shape."""
    points = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
    interpolated_points = interpolate_points(points, num_points=10)
    assert len(interpolated_points) == 10
    assert np.all(np.diff(interpolated_points[:, 0]) >= 0)  # Ensure x-coordinates are sorted

def test_interpolate_points_duplicate_x():
    """Test with duplicate x-coordinates."""
    points = np.array([[0, 0], [1, 1], [1, 2], [2, 2]])
    interpolated_points = interpolate_points(points, num_points=5)
    assert len(interpolated_points) == 5
    assert np.all(np.diff(interpolated_points[:, 0]) >= 0)

def test_interpolate_points_less_than_3():
    """Test with less than 3 points."""
    points = np.array([[0, 0], [1, 1]])
    interpolated_points = interpolate_points(points, num_points=10)
    assert np.array_equal(interpolated_points, points)

def test_interpolate_points_tolerance():
    """Test with points close to each other within tolerance."""
    points = np.array([[0, 0], [1, 1], [1 + 1e-9, 1.1], [2, 2]])
    interpolated_points = interpolate_points(points, num_points=5, tolerance=1e-8)
    assert len(interpolated_points) == 5
    assert np.all(np.diff(interpolated_points[:, 0]) >= 0)

def test_interpolate_points_unsorted():
    """Test with unsorted input points."""
    points = np.array([[2, 2], [0, 0], [1, 1]])
    interpolated_points = interpolate_points(points, num_points=5)
    assert len(interpolated_points) == 5
    assert np.allclose(interpolated_points[:, 0], np.linspace(0, 2, 5))

def test_interpolate_points_value_error():
    """Test when splprep raises a ValueError."""
    points = np.array([[0, 0], [1, 1], [1, 1]]) #causes a value error with splprep
    interpolated_points = interpolate_points(points, num_points=5)
    assert np.array_equal(interpolated_points, np.array([[0., 0.], [1., 1.]]))

def test_interpolate_points_num_points_zero():
    """Test when num_points is zero"""
    points = np.array([[0, 0], [1, 1], [2, 2]])
    interpolated_points = interpolate_points(points, num_points=0)
    assert len(interpolated_points) == 0