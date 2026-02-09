"""Backward-compatible re-export shim.

All utilities have been split into focused sub-modules:
  - misc.py          -- signal handling, hashing, tracing, Chinese detection, config merging
  - json_utils.py    -- JSON loading/dumping, code extraction
  - file_utils.py    -- file I/O, URL handling, file type detection
  - media_utils.py   -- image/audio/video base64 encoding/decoding
  - message_utils.py -- Message formatting, extraction, conversion

Prefer importing directly from the sub-module for new code.
"""

# Re-export logger so that ``from cat_agent.utils.utils import logger`` keeps working
from cat_agent.log import logger  # noqa: F401

from cat_agent.utils.misc import (  # noqa: F401
    append_signal_handler,
    get_local_ip,
    has_chinese_chars,
    hash_sha256,
    merge_generate_cfgs,
    print_traceback,
)

from cat_agent.utils.json_utils import (  # noqa: F401
    PydanticJSONEncoder,
    extract_code,
    json_dumps_compact,
    json_dumps_pretty,
    json_loads,
)

from cat_agent.utils.file_utils import (  # noqa: F401
    contains_html_tags,
    get_basename_from_url,
    get_content_type_by_head_request,
    get_file_type,
    is_http_url,
    is_image,
    read_text_from_file,
    sanitize_chrome_file_path,
    sanitize_windows_file_path,
    save_text_to_file,
    save_url_to_local_work_dir,
)

from cat_agent.utils.media_utils import (  # noqa: F401
    encode_audio_as_base64,
    encode_image_as_base64,
    encode_video_as_base64,
    load_image_from_base64,
    resize_image,
)

from cat_agent.utils.message_utils import (  # noqa: F401
    build_text_completion_prompt,
    extract_files_from_messages,
    extract_images_from_messages,
    extract_markdown_urls,
    extract_text_from_message,
    extract_urls,
    format_as_multimodal_message,
    format_as_text_message,
    get_last_usr_msg_idx,
    has_chinese_messages,
    rm_default_system,
)
