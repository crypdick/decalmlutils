from decalmlutils.training import generate_seed, seed_everything


def test_seed_everything():
    seed_everything(42)


def test_generate_seed():
    generate_seed()
