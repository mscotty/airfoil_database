import numpy as np
import logging
import pytest
from shapely.geometry import LineString, Point

from DASC500.xfoil.fix_airfoil_pointcloud_v2 import (  # Replace your_module with the actual module name
    detect_airfoil_format,
    detect_direction,
    reorder_partial_split_airfoil,
    reorder_split_airfoil,
    reorder_airfoil,
    check_closure,
    check_self_intersection,
    simplify_points,
    resolve_self_intersection,
    remove_duplicate_points,
    process_airfoil,
)

# Mock logging for tests
logging.basicConfig(level=logging.WARNING)

def test_detect_airfoil_format_split():
    points = np.array([[1, 0], [0, 0], [1, 1], [0, 1]])
    format_type, le_indices, te_indices = detect_airfoil_format(points)
    assert format_type == "split"
    assert len(le_indices) == 2
    assert len(te_indices) == 2

def test_detect_airfoil_format_partial_split():
    points = np.array([[1, 0], [0, 0], [1, 1], [0.5, 1]])
    format_type, le_indices, te_indices = detect_airfoil_format(points)
    assert format_type == "partial_split"
    assert len(le_indices) == 2

def test_detect_airfoil_format_closed():
    points = np.array([[0, 0], [1, 1], [2, 0], [1,-1], [0, 0]])
    format_type, le_idx, te_idx = detect_airfoil_format(points)
    assert format_type == "closed"
    assert le_idx == 2
    assert te_idx == 0

def test_detect_airfoil_format_unordered():
    points = np.array([[0, 0], [2, 0], [1, 1]])
    format_type, le_idx, te_idx = detect_airfoil_format(points)
    assert format_type == "unordered"
    assert le_idx == 1
    assert te_idx == 0

def test_detect_direction_le_to_te():
    points = np.array([[0, 0], [1, 1], [2, 0]])
    assert detect_direction(1, 0) == "TE_to_LE"

def test_detect_direction_te_to_le():
    points = np.array([[0, 0], [1, 1], [2, 0]])
    assert detect_direction(0, 1) == "LE_to_TE"

def test_reorder_partial_split_airfoil_le():
    points = np.array([[1, 0], [0, 0], [1, 1], [0.5, 1]])
    le_indices = np.array([0, 2])
    te_indices = np.array([])
    reordered = reorder_partial_split_airfoil(points, le_indices, te_indices)
    assert np.allclose(reordered[0], [1, 0])
    assert np.allclose(reordered[-1], [0.5, 1])

def test_reorder_partial_split_airfoil_te():
    points = np.array([[0, 0], [1, 1], [0, 1], [0.5, 1]])
    le_indices = np.array([])
    te_indices = np.array([0, 2])
    reordered = reorder_partial_split_airfoil(points, le_indices, te_indices)
    assert np.allclose(reordered[0], [0, 0])
    assert np.allclose(reordered[-1], [1, 1])

def test_reorder_split_airfoil():
    points = np.array([[2, 0.5], [1, 0], [0, 0], [2, 0.5], [1, 1], [0, 0]])
    le_indices = np.array([0, 2])
    te_indices = np.array([1, 3])
    reordered = reorder_split_airfoil(points, le_indices, te_indices)
    assert np.allclose(reordered[0], [2, 0.5])
    assert np.allclose(reordered[-1], [0, 0])

def test_reorder_airfoil_le_to_te():
    points = np.array([[0, 0], [2, 0], [1, 1]])
    reordered = reorder_airfoil(points, 1, 0)
    assert np.allclose(reordered[0], [2, 0])
    assert np.allclose(reordered[-1], [2, 0])

def test_reorder_airfoil_te_to_le():
    points = np.array([[0, 0], [2, 0], [1, 1]])
    reordered = reorder_airfoil(points, 0, 1)
    assert np.allclose(reordered[0], [0, 0])
    assert np.allclose(reordered[-1], [0, 0])

def test_check_closure_closed():
    points = np.array([[0, 0], [1, 1], [2, 0], [0, 0]])
    assert np.array_equal(check_closure(points), points)

def test_check_closure_open():
    points = np.array([[0, 0], [1, 1], [2, 0]])
    closed = check_closure(points)
    assert np.allclose(closed[-1], [0, 0])

def test_check_self_intersection_simple():
    points = np.array([[0, 0], [1, 1], [2, 0], [0, 0]])
    assert not check_self_intersection(points, "test")

def test_check_self_intersection_intersecting():
    points = np.array([[0, 0], [2, 2], [2, 0], [0, 2], [0, 0]])
    assert check_self_intersection(points, "test")

def test_simplify_points():
    points = np.array([[0, 0], [1, 1e-6], [2, 0], [3, 1]])
    simplified = simplify_points(points)
    assert len(simplified) == 3

def test_resolve_self_intersection_simple():
    points = np.array([[0, 0], [1, 1], [2, 0], [0, 0]])
    assert np.array_equal(resolve_self_intersection(points), points)

def test_resolve_self_intersection_intersecting():
    points = np.array([[0, 0], [2, 2], [2, 0], [0, 2], [0, 0]])
    resolved = resolve_self_intersection(points)
    assert LineString(resolved[:-1]).is_simple

def test_remove_duplicate_points():
    points = np.array([[0, 0], [0, 0], [1, 1], [2, 0], [2, 0]])
    unique = remove_duplicate_points(points)
    assert len(unique) == 4

def test_process_airfoil_unordered():
    points = np.array([[0, 0], [2, 0], [1, 1]])
    processed = process_airfoil("test", points)
    assert LineString(processed[:-1]).is_simple

def test_process_airfoil_split():
    points = np.array([[1, 0], [0, 0], [1, 1], [0, 1]])
    processed = process_airfoil("test", points)
    assert LineString(processed[:-1]).is_simple