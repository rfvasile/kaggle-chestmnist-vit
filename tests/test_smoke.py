import utils


def test_set_seed_is_reproducible():
    import random

    utils.set_seed(1234)
    first = random.random()
    utils.set_seed(1234)
    assert random.random() == first
