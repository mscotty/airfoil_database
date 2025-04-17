"""
Unit tests for the calculate_pointcloud_distance and calculate_min_distance_sum functions.

Uses pytest for testing.
"""

import numpy as np
import pytest

from DASC500.xfoil.calculate_distance import calculate_pointcloud_distance, calculate_min_distance_sum  # Replace your_module

def test_calculate_pointcloud_distance_equal():
    """Test with identical point clouds."""
    pc1 = np.array([[1, 2], [3, 4], [5, 6]])
    pc2 = np.array([[1, 2], [3, 4], [5, 6]])
    assert calculate_pointcloud_distance(pc1, pc2) == 0.0

def test_calculate_pointcloud_distance_shifted():
    """Test with point clouds shifted by a constant vector."""
    pc1 = np.array([[1, 2], [3, 4], [5, 6]])
    pc2 = np.array([[2, 3], [4, 5], [6, 7]])
    expected_distance = np.sqrt(2)
    assert np.isclose(calculate_pointcloud_distance(pc1, pc2), expected_distance)

def test_calculate_pointcloud_distance_different_sizes():
    """Test with point clouds of different sizes."""
    pc1 = np.array([[1, 2], [3, 4]])
    pc2 = np.array([[1, 2], [3, 4], [5, 6]])
    assert calculate_pointcloud_distance(pc1, pc2) == float('inf')

def test_calculate_pointcloud_distance_random():
    """Test with random point clouds."""
    pc1 = np.random.rand(10, 2)
    pc2 = np.random.rand(10, 2)
    distance = calculate_pointcloud_distance(pc1, pc2)
    assert distance >= 0.0

def test_calculate_min_distance_sum_equal():
    """Test calculate_min_distance_sum with identical point clouds."""
    pc1 = np.array([[1, 2], [3, 4]])
    pc2 = np.array([[1, 2], [3, 4]])
    assert calculate_min_distance_sum(pc1, pc2) == 0.0

def test_calculate_min_distance_sum_shifted():
    """Test calculate_min_distance_sum with shifted point clouds."""
    pc1 = np.array([[1, 2], [3, 4]])
    pc2 = np.array([[2, 3], [4, 5]])
    expected_distance = 2 * np.sqrt(2)
    assert np.isclose(calculate_min_distance_sum(pc1, pc2), expected_distance)

def test_calculate_min_distance_sum_varying_distances():
    """Test calculate_min_distance_sum with varying minimum distances."""
    pc1 = np.array([[0, 0], [2, 2]])
    pc2 = np.array([[1, 0], [0, 1], [3, 2]])
    expected_distance = 1.0 + 1.0
    assert np.isclose(calculate_min_distance_sum(pc1, pc2), expected_distance)

def test_calculate_min_distance_sum_empty_pc1():
    """Test calculate_min_distance_sum with an empty pc1."""
    pc1 = np.array([])
    pc2 = np.array([[1, 2], [3, 4]])
    assert calculate_min_distance_sum(pc1, pc2) == 0.0

def test_calculate_min_distance_sum_empty_pc2():
    """Test calculate_min_distance_sum with an empty pc2."""
    pc1 = np.array([[1, 2], [3, 4]])
    pc2 = np.array([])
    expected_distance = float('inf')
    assert calculate_min_distance_sum(pc1, pc2) == float('inf')

def test_calculate_min_distance_sum_empty_both():
    """Test calculate_min_distance_sum with both empty point clouds."""
    pc1 = np.array([])
    pc2 = np.array([])
    assert calculate_min_distance_sum(pc1, pc2) == float('inf')