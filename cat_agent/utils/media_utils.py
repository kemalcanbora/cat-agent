"""Image, audio and video encoding / decoding utilities."""

import base64
from io import BytesIO
from typing import Union

from cat_agent.log import logger


def encode_image_as_base64(path: str, max_short_side_length: int = -1) -> str:
    from PIL import Image
    image = Image.open(path)

    if (max_short_side_length > 0) and (min(image.size) > max_short_side_length):
        ori_size = image.size
        image = resize_image(image, short_side_length=max_short_side_length)
        logger.debug(f'Image "{path}" resized from {ori_size} to {image.size}.')

    image = image.convert(mode='RGB')
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(buffered.getvalue()).decode('utf-8')


def encode_audio_as_base64(path: str) -> str:
    with open(path, 'rb') as audio_file:
        return 'data:;base64,' + base64.b64encode(audio_file.read()).decode('utf-8')


def encode_video_as_base64(path: str) -> str:
    with open(path, 'rb') as video_file:
        return 'data:;base64,' + base64.b64encode(video_file.read()).decode('utf-8')


def load_image_from_base64(image_base64: Union[bytes, str]):
    from PIL import Image
    image = Image.open(BytesIO(base64.b64decode(image_base64)))
    image.load()
    return image


def resize_image(img, short_side_length: int = 1080):
    from PIL import Image
    assert isinstance(img, Image.Image)

    width, height = img.size
    if width <= height:
        new_width = short_side_length
        new_height = int((short_side_length / width) * height)
    else:
        new_height = short_side_length
        new_width = int((short_side_length / height) * width)

    return img.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
