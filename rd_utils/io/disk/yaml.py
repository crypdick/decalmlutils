import yaml
from beartype import beartype


@beartype
def read_yaml_from_disk(fpath: str) -> dict:
    with open(fpath, "r") as f:
        yaml_as_dict = yaml.safe_load(f)
    return yaml_as_dict
