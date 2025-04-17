"""
Unit tests for the leading_edge_radius function in DASC500.formulas.airfoil.

Uses pytest for testing.
"""

import numpy as np
import pytest

from DASC500.formulas.airfoil.compute_LE_radius import leading_edge_radius  # Adjust import as needed

def test_leading_edge_radius_simple_curve():
    """Test with a simple parabolic leading edge."""
    points = np.array([
        [0, 0],
        [0.01, 0.1],
        [0.02, 0.2],
        [0.03, 0.3],
        [0.04, 0.4],
        [0.05, 0.5],
        [0.1, 1],
    ])
    radius = leading_edge_radius(points)
    # The expected radius depends on the curve's exact shape,
    # so we use a tolerance for comparison.
    assert np.isclose(radius, 0.05, atol=0.01)

def test_leading_edge_radius_horizontal_leading_edge():
    """Test with a flat (horizontal) leading edge. Should give a large radius."""
    points = np.array([
        [0, 0],
        [0.01, 0],
        [0.02, 0],
        [0.03, 0],
        [0.04, 0],
        [0.05, 0],
        [0.1, 0],
    ])
    radius = leading_edge_radius(points)
    assert radius > 1000  # Expect a very large radius

def test_leading_edge_radius_sharp_leading_edge():
    """Test with a sharp (near-vertical) leading edge. Should give a small radius."""
    points = np.array([
        [0, 0],
        [0.01, 0.01],
        [0.02, 0.02],
        [0.03, 0.03],
        [0.04, 0.04],
        [0.05, 0.05],
        [0.1, 0.1],
    ])
    radius = leading_edge_radius(points)
    assert np.isclose(radius, 0.05, atol=0.01)

def test_leading_edge_radius_noisy_data():
    """Test with noisy data around the leading edge."""
    points = np.array([
        [0, 0.01],
        [0.01, 0.11],
        [0.02, 0.19],
        [0.03, 0.32],
        [0.04, 0.39],
        [0.05, 0.51],
        [0.1, 1],
    ])
    radius = leading_edge_radius(points)
    assert np.isclose(radius, 0.05, atol=0.02)

def test_leading_edge_radius_non_uniform_spacing():
    """Test with non-uniformly spaced leading edge points."""
    points = np.array([
        [0, 0],
        [0.01, 0.1],
        [0.03, 0.3],
        [0.04, 0.4],
        [0.049, 0.49],
        [0.05, 0.5],
        [0.1, 1],
    ])
    radius = leading_edge_radius(points)
    assert np.isclose(radius, 0.05, atol=0.01)

def test_leading_edge_radius_small_point_set():
    """Test with a small set of points around the leading edge."""
    points = np.array([
        [0, 0],
        [0.01, 0.1],
        [0.02, 0.2],
    ])
    radius = leading_edge_radius(points)
    assert np.isclose(radius, 0.05, atol=0.01)

def test_leading_edge_radius_negative_x():
    """Test with negative x-coordinates."""
    points = np.array([
        [-0.05,0.5],
        [-0.04, 0.4],
        [-0.03, 0.3],
        [-0.02, 0.2],
        [-0.01, 0.1],
        [0,0],
    ])
    radius = leading_edge_radius(points)
    assert np.isclose(radius, 0.05, atol=0.01)

def test_leading_edge_radius_single_point():
    """Test with only one point within the 0.05 delta."""
    points = np.array([
        [0, 0],
        [0.1, 1],
    ])
    radius = leading_edge_radius(points)
    assert np.isnan(radius) #polyfit will fail, and return nan.