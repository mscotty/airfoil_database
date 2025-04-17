"""
Unit tests for the thickness_to_chord_ratio function.

Uses pytest for testing.
"""

import numpy as np
import pytest

from DASC500.formulas.airfoil.compute_thickness_to_chord_ratio import thickness_to_chord_ratio  # Replace your_module

def test_thickness_to_chord_ratio_positive():
    """Test with positive thickness and chord."""
    thickness = np.array([0.1, 0.2, 0.3, 0.25])
    chord_length = 1.0
    expected_ratio = 0.3
    assert thickness_to_chord_ratio(thickness, chord_length) == expected_ratio

def test_thickness_to_chord_ratio_non_integer():
    """Test with non-integer thickness and chord."""
    thickness = np.array([0.15, 0.25, 0.35, 0.3])
    chord_length = 1.5
    expected_ratio = 0.35 / 1.5
    assert thickness_to_chord_ratio(thickness, chord_length) == expected_ratio

def test_thickness_to_chord_ratio_large_values():
    """Test with large thickness and chord values."""
    thickness = np.array([10, 20, 30, 25])
    chord_length = 100.0
    expected_ratio = 30.0 / 100.0
    assert thickness_to_chord_ratio(thickness, chord_length) == expected_ratio

def test_thickness_to_chord_ratio_small_values():
    """Test with small thickness and chord values."""
    thickness = np.array([0.01, 0.02, 0.03, 0.025])
    chord_length = 0.1
    expected_ratio = 0.03 / 0.1
    assert thickness_to_chord_ratio(thickness, chord_length) == expected_ratio

def test_thickness_to_chord_ratio_zero_thickness():
    """Test with zero thickness."""
    thickness = np.array([0.0, 0.0, 0.0])
    chord_length = 1.0
    expected_ratio = 0.0
    assert thickness_to_chord_ratio(thickness, chord_length) == expected_ratio

def test_thickness_to_chord_ratio_zero_chord():
    """Test with zero chord. Should raise a ZeroDivisionError."""
    thickness = np.array([0.1, 0.2, 0.3])
    chord_length = 0.0
    with pytest.raises(ZeroDivisionError):
        thickness_to_chord_ratio(thickness, chord_length)

def test_thickness_to_chord_ratio_negative_thickness():
    """Test with negative thickness. Should return a negative ratio."""
    thickness = np.array([-0.1, -0.2, -0.3])
    chord_length = 1.0
    expected_ratio = -0.1 #Corrected expected value
    assert thickness_to_chord_ratio(thickness, chord_length) == expected_ratio

def test_thickness_to_chord_ratio_negative_chord():
    """Test with negative chord. Should return a negative ratio."""
    thickness = np.array([0.1, 0.2, 0.3])
    chord_length = -1.0
    expected_ratio = -0.3
    assert thickness_to_chord_ratio(thickness, chord_length) == expected_ratio

def test_thickness_to_chord_ratio_empty_thickness():
    """Test with an empty thickness array. Should raise a ValueError."""
    thickness = np.array([])
    chord_length = 1.0
    with pytest.raises(ValueError):
        thickness_to_chord_ratio(thickness, chord_length)

def test_thickness_to_chord_ratio_single_thickness():
    """Test with a single thickness value."""
    thickness = np.array([0.2])
    chord_length = 1.0
    expected_ratio = 0.2
    assert thickness_to_chord_ratio(thickness, chord_length) == expected_ratio