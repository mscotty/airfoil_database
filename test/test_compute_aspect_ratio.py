# -*- coding: utf-8 -*-
"""
Unit tests for the calculate_aspect_ratio function.

Uses pytest for testing.
"""

import pytest

from DASC500.formulas.airfoil.compute_aspect_ratio import calculate_aspect_ratio  # Replace your_module

def test_calculate_aspect_ratio_positive():
    """Test with positive span and chord."""
    span = 10.0
    chord = 2.0
    expected_ratio = 5.0
    assert calculate_aspect_ratio(span, chord) == expected_ratio

def test_calculate_aspect_ratio_non_integer():
    """Test with non-integer span and chord."""
    span = 7.5
    chord = 1.5
    expected_ratio = 5.0
    assert calculate_aspect_ratio(span, chord) == expected_ratio

def test_calculate_aspect_ratio_large_values():
    """Test with large span and chord values."""
    span = 1000.0
    chord = 100.0
    expected_ratio = 10.0
    assert calculate_aspect_ratio(span, chord) == expected_ratio

def test_calculate_aspect_ratio_small_values():
    """Test with small span and chord values."""
    span = 0.5
    chord = 0.1
    expected_ratio = 5.0
    assert calculate_aspect_ratio(span, chord) == expected_ratio

def test_calculate_aspect_ratio_zero_span():
    """Test with zero span."""
    span = 0.0
    chord = 2.0
    expected_ratio = 0.0
    assert calculate_aspect_ratio(span, chord) == expected_ratio

def test_calculate_aspect_ratio_zero_chord():
    """Test with zero chord. Should raise a ZeroDivisionError."""
    span = 10.0
    chord = 0.0
    with pytest.raises(ZeroDivisionError):
        calculate_aspect_ratio(span, chord)

def test_calculate_aspect_ratio_negative_span():
    """Test with negative span. Should return a negative aspect ratio."""
    span = -10.0
    chord = 2.0
    expected_ratio = -5.0
    assert calculate_aspect_ratio(span, chord) == expected_ratio

def test_calculate_aspect_ratio_negative_chord():
    """Test with negative chord. Should return a negative aspect ratio."""
    span = 10.0
    chord = -2.0
    expected_ratio = -5.0
    assert calculate_aspect_ratio(span, chord) == expected_ratio

def test_calculate_aspect_ratio_negative_both():
    """Test with negative span and chord. Should return a positive aspect ratio."""
    span = -10.0
    chord = -2.0
    expected_ratio = 5.0
    assert calculate_aspect_ratio(span, chord) == expected_ratio