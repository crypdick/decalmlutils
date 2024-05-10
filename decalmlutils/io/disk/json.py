import io
import json

from beartype import beartype
from beartype.typing import Dict, List, Union

from decalmlutils.io.context import safe_open
from decalmlutils.io.disk.misc import create_dir_if_not_exists, read_bytes_from_disk


@beartype
def read_json_from_disk(fpath: str) -> dict:
    json_bytes = read_bytes_from_disk(fpath)
    json_as_dict: dict = json.load(io.BytesIO(json_bytes))
    return json_as_dict


@beartype
def write_json_to_disk(json_as_dict: Union[List, Dict], outpath: str) -> None:
    create_dir_if_not_exists(outpath)
    with safe_open(outpath, "w+") as f:
        # indent for pretty output
        json.dump(json_as_dict, f, indent=2, sort_keys=False)
