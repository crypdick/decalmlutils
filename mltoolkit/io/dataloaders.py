import logging
import random

import numpy as np
import torch
from beartype.typing import Any, Callable, Optional, Tuple, Union
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from mltoolkit.io.aws import get_aws_client
from mltoolkit.io.aws.s3_local import check_get_img, s3url_to_local_mirror
from mltoolkit.io.disk.img import create_dummy_img, read_img_from_disk
from mltoolkit.io.disk.misc import delete_file_if_exists

TILE_PIXEL_WIDTH = 256
MOCK_IMG_SIZE = 256

MOCK_NUM_CLASSES = 10
CORRUPT_CHANCE = 1.0 / 200

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        s3urls: np.ndarray,
        labels: np.ndarray,
        version: str,
        use_fp16: bool,
        defective_idx: int,
        corrupt_transform: Callable,
        randomly_corrupt: bool,
        transform: Optional[Any] = None,
    ):
        self.s3urls = s3urls
        self.version = version
        self.use_fp16 = use_fp16
        self.defective_idx = defective_idx
        self.corrupt_transform = corrupt_transform
        self.randomly_corrupt = randomly_corrupt
        self.transform = transform
        self.encoding_len = np.shape(labels)[-1]
        self.num_failures = 0

        # process all labels at once instead of one at a time
        self.labels = torch.from_numpy(labels)
        if self.use_fp16:
            self.labels = self.labels.half()

        self.__post_init__()

    def __post_init__(self):
        pass

    def __len__(self) -> int:
        return len(self.s3urls)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        label = self._get_label(idx)
        image = self._get_img(idx)
        image, label = self._randomly_corrupt(image, label)
        if self.use_fp16:
            # note: if using torch transforms v2, just make the transform directly cast to half
            image = image.half()

        return image, label

    def _get_img(self, img_s3url: str) -> Union[np.ndarray, Tensor]:
        raise NotImplementedError("This method must be implemented in a subclass.")

    def _get_label(self, idx: int) -> Tensor:
        return self.labels[idx]

    def _randomly_corrupt(
        self, image: np.ndarray, label: Tensor
    ) -> Tuple[np.ndarray, Tensor]:
        """
        Apply a 1 in CORRUPT_CHANCE chance to corrupt the image. If corrupted, convert the label to
        DEFECTIVE_PROCESSING.

        Otherwise, apply the usual transform and do not change the label to DEFECTIVE_PROCESSING.
        """
        if self.randomly_corrupt and random.random() < CORRUPT_CHANCE:
            image = self.corrupt_transform(image)
            label = torch.zeros_like(label)  # convert the label to DEFECTIVE_PROCESSING
            label[self.defective_idx] = 1
        else:
            # apply the usual transform and do not change the label to DEFECTIVE_PROCESSING
            image = self.transform(image)
        return image, label


class MockDataset(BaseDataset):
    def __len__(self) -> int:
        return MOCK_NUM_CLASSES

    def _get_label(self, idx) -> Tensor:
        # create a torch tensor with random 1s and 0s
        # this is important so that the AUC score has both positive and negative examples for each class,
        # otherwise it'll be undefined
        return torch.randint(0, 2, (MOCK_NUM_CLASSES,))

    def _get_img(self, idx) -> np.ndarray:
        return create_dummy_img(img_size=MOCK_IMG_SIZE, mode="RGB", as_type="array")


class PreCachedDataset(BaseDataset):
    """
    Pytorch Dataset which reads images from disk.

    This is faster than S3ImageDataset, but requires that the images be pre-cached.
    """

    def __post_init__(self):
        self.local_fpaths = [s3url_to_local_mirror(s3url) for s3url in self.s3urls]

    def _get_img(self, idx) -> np.ndarray:
        img_local_path = self.local_fpaths[idx]
        try:
            image = read_img_from_disk(img_local_path, mode="RGB", as_type="array")
        except BaseException as e:
            delete_file_if_exists(img_local_path)
            # this makes the code more robust to missing images. if the local img is corrupted, this function
            # will delete it and fetch it again. note: this will create a new S3 client for each failed image,
            # which is slow. to pre-cache at max speed, see `precache_training_data()`
            self.num_failures += 1
            logger.error(
                f"could not read image from disk: {img_local_path} with {e}. Falling back to S3 download."
                f"num_failures={self.num_failures}"
            )
            payload = check_get_img(
                self.s3urls[idx],
                check_cache=True,
                store_copy=True,
                mode="RGB",
                as_type="array",
            )
            image = payload.data

        return image


class S3ImageDataset(BaseDataset):
    """
    Pytorch Dataset which fetches images from S3.

    This class is slower than PreCachedDataset, but it is more robust since it will fetch any missing images from S3.
    It will also check the cache before fetching from S3, and if it does detect a cache miss, it will fetch the image
    and store it in the cache for future epochs.

    If all the images are pre-cached, this class will be around 8.5% slower than PreCachedDataset due to the extra
    S3 calls and cache checks.

    Based on: https://stackoverflow.com/a/54096125/4212158
    """

    def __post_init__(self):
        self.s3c = None  # wait to instantiate until S3ImageDataset is pickled and distributed to nodes

    def _get_img(self, idx) -> np.ndarray:
        img_s3url = self.s3urls[idx]
        # by this point we should be running inside a worker's process
        # and thus can safely attach s3 objects to class instance
        if not self.s3c:
            self.s3c = get_aws_client("s3")

        payload = check_get_img(
            img_s3url,
            check_cache=True,
            store_copy=True,
            mode="RGB",
            as_type="array",
            client=self.s3c,
        )
        image = payload.data
        return image


class FastPreCachedDataset(BaseDataset):
    """
    Like PreCachedDataset, but tries to be as fast as possible.
    """

    def __post_init__(self):
        self.local_fpaths = [s3url_to_local_mirror(s3url) for s3url in self.s3urls]
        self.label_singleton = torch.Tensor([0.0])

    def _get_img(self, idx) -> Image.Image:
        img_local_path = self.local_fpaths[idx]
        # simplified logic for fastest img loading
        image = Image.open(img_local_path, formats=["png"]).convert("RGB")

        return image

    def _get_label(self, idx) -> Tensor:
        return self.label_singleton
