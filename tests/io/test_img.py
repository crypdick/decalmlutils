import numpy as np
from PIL import Image

from rd_utils.io.disk.img import read_img_from_disk

# create fixtures for temporary RGBA png imgs saved on disk
# one is just fully opaque white pixels
# one is fully transparent
# one is 50% opaque
IMG_SIZE = 4


def test_read_img_from_disk(tmp_path):
    # for each img, test that both as_array=True and as_array=False return the same result

    # fully opaque, random pixels
    init_image = np.random.randint(0, 255, size=(IMG_SIZE, IMG_SIZE, 4), dtype=np.uint8)
    init_image = Image.fromarray(init_image)
    tmp_fpath = f"{tmp_path}/half_opaque.png"
    init_image.save(tmp_fpath)

    img_pil = read_img_from_disk(tmp_fpath, "RGBA", as_type="image")
    assert np.array_equal(img_pil, np.array(init_image))

    img_np = read_img_from_disk(tmp_fpath, "RGBA", as_type="array")
    assert np.array_equal(img_np, np.array(init_image))

    img_tensor = read_img_from_disk(tmp_fpath, "RGBA", as_type="tensor")
    # gotcha: tensor is channels_first, so we need to transpose
    img_tensor = img_tensor.permute(1, 2, 0)
    # gotcha2: ToTensor converts to [0, 1] range, need to convert back to [0, 255]
    img_tensor = img_tensor * 255
    assert np.array_equal(img_tensor.numpy(), np.array(init_image))

    # check that RGB mode works
    img_pil = read_img_from_disk(tmp_fpath, "RGB", as_type="image")
    assert np.array_equal(img_pil, np.array(init_image.convert("RGB")))

    img_np = read_img_from_disk(tmp_fpath, "RGB", as_type="array")
    assert np.array_equal(img_np, np.array(init_image.convert("RGB")))

    img_tensor = read_img_from_disk(tmp_fpath, "RGB", as_type="tensor")
    # gotcha: tensor is channels_first, so we need to transpose
    img_tensor = img_tensor.permute(1, 2, 0)
    # gotcha2: ToTensor converts to [0, 1] range, need to convert back to [0, 255]
    img_tensor = img_tensor * 255
    assert np.array_equal(img_tensor.numpy(), np.array(init_image.convert("RGB")))
