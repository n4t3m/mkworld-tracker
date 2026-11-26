from mktracker.detection.result_grid import rank_origin


def test_rank_origin_uses_consistent_digits_across_missing_rows():
    assert rank_origin([(0, 8), (1, None), (2, 10), (4, 12)],
                       min_support=2) == (8, 3)


def test_rank_origin_requires_independent_anchors_when_requested():
    assert rank_origin([(0, None), (1, 9)], min_support=2) is None
    assert rank_origin([(0, None), (1, 9)]) == (8, 1)
