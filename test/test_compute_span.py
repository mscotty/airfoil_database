"""
Unit tests for the calculate_span function in airfoil_span.py.

Uses pytest for testing.
"""

import math
import numpy as np
import pytest

from DASC500.formulas.airfoil.compute_span import calculate_span  # Assuming your file is named airfoil_span.py

def test_calculate_span_simple_line():
    """Test with a simple straight line."""
    points = [(0, 0), (1, 1)]
    expected_length = math.sqrt(2)
    assert math.isclose(calculate_span(points), expected_length)

def test_calculate_span_triangle():
    """Test with a triangle shape."""
    points = [(0, 0), (1, 0), (0, 1), (0, 0)]
    expected_length = 2 + math.sqrt(2)
    assert math.isclose(calculate_span(points), expected_length)

def test_calculate_span_square():
    """Test with a square shape."""
    points = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    expected_length = 4
    assert math.isclose(calculate_span(points), expected_length)

def test_calculate_span_empty_list():
    """Test with an empty list of points."""
    points = []
    assert calculate_span(points) == 0.0

def test_calculate_span_single_point():
    """Test with a single point."""
    points = [(0, 0)]
    assert calculate_span(points) == 0.0

def test_calculate_span_non_uniform_spacing():
    """Test with non-uniformly spaced points."""
    points = [(0, 0), (1, 0), (1.5, 0.5), (2, 1)]
    expected_length = 1 + math.sqrt(0.5) + math.sqrt(0.5)
    assert math.isclose(calculate_span(points), expected_length)

def test_calculate_span_numpy_array():
    """Test with numpy array input."""
    points = np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    expected_length = 4
    assert math.isclose(calculate_span(points), expected_length)

def test_calculate_span_negative_coordinates():
    """Test with negative coordinates."""
    points = [(-1, -1), (0, 0), (1, 1)]
    expected_length = 2 * math.sqrt(2)
    assert math.isclose(calculate_span(points), expected_length)