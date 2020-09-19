import pytest

from bask.init import sb_sequence


def test_sb_sequence():
    random_state = 1
    x = sb_sequence(n=5, d=1, random_state=random_state)
    assert x.shape == (5, 1)

    existing = [(0.5, 0.5)]
    x = sb_sequence(n=5, d=2, existing_points=existing, random_state=random_state)
    assert x.shape == (5, 2)

    existing = [(0.5, 0.5)]
    with pytest.raises(ValueError):
        sb_sequence(n=1, d=2, existing_points=existing, random_state=random_state)
