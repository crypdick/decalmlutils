import io

import geojson
from beartype import beartype
from beartype.typing import Union
from mltoolkit.io.context import safe_open
from mltoolkit.io.disk.misc import create_dir_if_not_exists, read_bytes_from_disk

# Geojson default precision is way too low for our purposes
geojson.geometry.Geometry.__init__.__defaults__ = (None, False, 8)


@beartype
def read_geojson_from_disk(
    fpath: str,
) -> Union[geojson.Feature, geojson.FeatureCollection]:
    """
    Return a list of features from a feature collection or geojson.
    """
    bytes_arr = read_bytes_from_disk(fpath)
    feature_collection = geojson.load(io.BytesIO(bytes_arr))
    # return the entire feature collection instead of a list of features -returning only the latter, we lose some keys.
    return feature_collection


@beartype
def write_geojson_to_disk(
    features: Union[geojson.FeatureCollection, geojson.Feature], fpath: str
) -> None:
    """
    Args:
          features (geojson.FeatureCollection):


          fpath (string):
                      The filename of the geojson to be written.


    """
    # assert features.is_valid, AssertionError(features.errors())  # 22-04-05 was getting AttributeError for this
    create_dir_if_not_exists(fpath)
    with safe_open(fpath, "w") as f:
        geojson.dump(features, f, indent=4)
