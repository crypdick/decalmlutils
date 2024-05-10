import io
import logging

import numpy as np
import PIL
import torchvision.transforms.functional as TF
from beartype import beartype
from beartype.typing import Literal, Union
from PIL import Image
from torch import Tensor

from decalmlutils.io.context import safe_open
from decalmlutils.io.disk.misc import create_dir_if_not_exists

# note: do not use get_cloudwatch_logger() here. it will cause Ray serialization errors
logger = logging.getLogger(__name__)
DUMMY_IMG_FILL_VALUE = 42


@beartype
def img_bytes_to_Image(img_bytes: bytes, mode: str) -> Image.Image:
    """
    Converts a bytes object of a PNG image to a PIL Image object. The image is converted to RGB.

    Args:
        img_bytes: a bytes array representing a PNG image

    Returns:
        PIL Image object (uint8) in RGB format
    """
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert(mode)

    return img


@beartype
def convert_image_type(
    image: Image.Image, as_type: Literal["array", "tensor", "image"]
) -> Union[Image.Image, np.ndarray, Tensor]:
    if as_type == "image":
        pass
    if as_type == "tensor":
        # to_tensor maps values to [0, 1] if they are uint8
        # it also converts channels-last format to CHW format
        # note that previously, we used albumentations ToTensorV2(), which is not forward compatible
        # with torchvision ToTensor()
        image = TF.to_tensor(image)
    elif as_type == "array":
        # gotcha: if we use np.asarray, we will share memory with the original image
        # this is faster, but causes problems with setting flags (e.g. not writeable)
        # so, we have to use np.array() instead, which sucks bc it is slower
        image = np.array(image, copy=True)

    return image


@beartype
def read_img_from_disk(
    fpath: str, mode: Literal["RGB", "RGBA"] = "RGB", as_type: str = "image"
) -> Union[Image.Image, np.ndarray, Tensor]:
    """
    Read an image from disk.

    Args:
        fpath: path to local img file

    Returns:
        img: image as a PIL Image object (uint8)
    """
    img = Image.open(fpath).convert(mode)

    # img = img.copy()  # handles tensor not writeable errors. about 20% slower, though
    img = convert_image_type(img, as_type=as_type)

    return img


@beartype
def write_img_to_disk(img: Image.Image, out_fpath: str) -> None:
    create_dir_if_not_exists(out_fpath)
    try:
        with safe_open(out_fpath, "wb", ".png") as f:
            img.save(f)
    except Exception as e:
        logger.exception(
            f"writing image failed, output destination {out_fpath}, exception {e}", e
        )
        raise e


@beartype
def create_dummy_img(
    mode: str,
    as_type: str,
    img_size: int = 8,
    transparent: bool = False,
) -> Union[PIL.Image.Image, np.ndarray, Tensor]:
    # height and width of 8 b/c Dropout aug has those as min sizes
    # note: if we change the img_size, we need to update add_white_branches()
    # note: cannot use fill value of 0, since we are now dropping completely transparent images during inference
    dummy_img = PIL.Image.new(
        mode="RGBA",
        size=(img_size, img_size),
        color=(
            DUMMY_IMG_FILL_VALUE,
            DUMMY_IMG_FILL_VALUE,
            DUMMY_IMG_FILL_VALUE,
            0 if transparent else 255,
        ),
    )
    dummy_img = dummy_img.convert(mode)
    dummy_img = convert_image_type(dummy_img, as_type)

    return dummy_img
