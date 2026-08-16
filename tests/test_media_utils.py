# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for cat_agent.utils.media_utils."""

import base64
from io import BytesIO

from PIL import Image

from cat_agent.utils.media_utils import (
    encode_audio_as_base64,
    encode_image_as_base64,
    encode_video_as_base64,
    load_image_from_base64,
    resize_image,
)


def _make_png(path, size=(200, 100)):
    img = Image.new('RGB', size, color=(10, 20, 30))
    img.save(path, format='PNG')


def test_encode_and_load_image(tmp_path):
    path = tmp_path / 'x.png'
    _make_png(path)
    data_url = encode_image_as_base64(str(path))
    assert data_url.startswith('data:image/jpeg;base64,')
    b64 = data_url.split(',', 1)[1]
    img = load_image_from_base64(b64)
    assert img.size[0] > 0


def test_encode_image_resizes_short_side(tmp_path):
    path = tmp_path / 'big.png'
    _make_png(path, size=(400, 200))
    data_url = encode_image_as_base64(str(path), max_short_side_length=100)
    b64 = data_url.split(',', 1)[1]
    img = load_image_from_base64(b64)
    assert min(img.size) == 100


def test_resize_image_landscape():
    img = Image.new('RGB', (200, 100))
    out = resize_image(img, short_side_length=50)
    assert out.size == (100, 50)


def test_encode_audio_and_video(tmp_path):
    audio = tmp_path / 'a.bin'
    video = tmp_path / 'v.bin'
    audio.write_bytes(b'abc')
    video.write_bytes(b'def')
    assert encode_audio_as_base64(str(audio)).endswith(base64.b64encode(b'abc').decode())
    assert encode_video_as_base64(str(video)).endswith(base64.b64encode(b'def').decode())
