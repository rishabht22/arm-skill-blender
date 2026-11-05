import pytest


@pytest.fixture
def library():
    """Small valid Panda configurations for execution tests; no dataset required."""
    initial = [0., .3927, 0., -1.9635, 0., 2.3562, .7854, .04, .04]
    return {"arms": [{"initial": initial, "approach": initial[:7],
                      "lift": [0., 0., 0., -2.2, 0., 2.2, .7854]} for _ in range(2)]}
