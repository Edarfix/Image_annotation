import math

import numpy as np
import pytest

from ovannot.utils.boxes import (
    box_area,
    clip_xyxy,
    expand_xyxy,
    iou_matrix,
    is_valid_xyxy,
    nms,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_xywh_xyxy_round_trip():
    assert xywh_to_xyxy((10, 20, 30, 40)) == (10.0, 20.0, 40.0, 60.0)
    assert xyxy_to_xywh((10, 20, 40, 60)) == (10.0, 20.0, 30.0, 40.0)
    box = (1.5, 2.25, 7.75, 9.0)
    assert xywh_to_xyxy(xyxy_to_xywh(box)) == pytest.approx(box)


def test_area_and_clip():
    assert box_area((0, 0, 10, 5)) == 50
    assert box_area((5, 5, 2, 2)) == 0  # inverted box has no area
    assert clip_xyxy((-5, -1, 120, 50), width=100, height=40) == (0.0, 0.0, 100.0, 40.0)


@pytest.mark.parametrize(
    ("box", "ok"),
    [
        ((0, 0, 10, 10), True),
        ((10, 0, 10, 10), False),  # zero width
        ((0, 0, math.nan, 10), False),
        ((0, 0, 10), False),
        ((-0.5, 0, 100.5, 10), True),  # within the 1 px tolerance
        ((-5, 0, 10, 10), False),
        ((0, 0, 10, 45), False),  # below the image
    ],
)
def test_is_valid_xyxy(box, ok):
    assert is_valid_xyxy(box, width=100, height=40) is ok


def test_expand_is_clipped():
    assert expand_xyxy((10, 10, 20, 30), margin=0.5, width=100, height=100) == (
        5.0,
        0.0,
        25.0,
        40.0,
    )
    assert expand_xyxy((0, 0, 10, 10), margin=1.0, width=15, height=15) == (0.0, 0.0, 15.0, 15.0)


def test_iou_matrix_known_values():
    a = [[0, 0, 10, 10], [0, 0, 10, 10]]
    b = [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]
    expected = np.array([[1.0, 50 / 150, 0.0], [1.0, 50 / 150, 0.0]])
    assert iou_matrix(a, b) == pytest.approx(expected)
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_iou_degenerate_boxes_do_not_divide_by_zero():
    assert iou_matrix([[0, 0, 0, 0]], [[0, 0, 0, 0]]) == pytest.approx(np.zeros((1, 1)))


def test_nms_keeps_best_of_overlapping_boxes():
    boxes = [[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]]
    scores = [0.6, 0.9, 0.5]
    assert nms(boxes, scores, iou_threshold=0.5) == [1, 2]
    assert nms(boxes, scores, iou_threshold=0.95) == [1, 0, 2]
    assert nms(np.zeros((0, 4)), [], iou_threshold=0.5) == []
