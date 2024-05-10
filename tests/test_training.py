from decalmlutils.training import create_transform, generate_seed, seed_everything


def test_seed_everything():
    seed_everything(42)


def test_generate_seed():
    generate_seed()


def test_create_transform_eval():
    mean = [0.5, 0.5, 0.5]
    stddev = [0.5, 0.5, 0.5]
    transform = create_transform("eval", mean, stddev, version="v1")
    assert callable(transform), "The returned object is not callable"
