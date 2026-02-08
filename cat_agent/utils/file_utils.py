"""File-system and URL helpers: reading, downloading, type detection."""

import os
import re
import shutil
import time
import urllib.parse
from typing import Literal

import requests

from cat_agent.log import logger
from cat_agent.utils.misc import print_traceback


# ---------------------------------------------------------------------------
# Basename / URL helpers
# ---------------------------------------------------------------------------


def get_basename_from_url(path_or_url: str) -> str:
    if re.match(r'^[A-Za-z]:\\', path_or_url):
        path_or_url = path_or_url.replace('\\', '/')

    basename = urllib.parse.urlparse(path_or_url).path
    basename = os.path.basename(basename)
    basename = urllib.parse.unquote(basename)
    basename = basename.strip()

    if not basename:
        basename = [x.strip() for x in path_or_url.split('/') if x.strip()][-1]

    return basename


def is_http_url(path_or_url: str) -> bool:
    return path_or_url.startswith('https://') or path_or_url.startswith('http://')


def is_image(path_or_url: str) -> bool:
    filename = get_basename_from_url(path_or_url).lower()
    return any(filename.endswith(ext) for ext in ('jpg', 'jpeg', 'png', 'webp'))


# ---------------------------------------------------------------------------
# Path sanitisation
# ---------------------------------------------------------------------------


def sanitize_chrome_file_path(file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path

    new_path = urllib.parse.urlparse(file_path)
    new_path = urllib.parse.unquote(new_path.path)
    new_path = sanitize_windows_file_path(new_path)
    if os.path.exists(new_path):
        return new_path

    return sanitize_windows_file_path(file_path)


def sanitize_windows_file_path(file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path

    win_path = file_path
    if win_path.startswith('/'):
        win_path = win_path[1:]
    if os.path.exists(win_path):
        return win_path

    if re.match(r'^[A-Za-z]:/', win_path):
        wsl_path = f'/mnt/{win_path[0].lower()}/{win_path[3:]}'
        if os.path.exists(wsl_path):
            return wsl_path

    win_path = win_path.replace('/', '\\')
    if os.path.exists(win_path):
        return win_path

    return file_path


# ---------------------------------------------------------------------------
# Download / copy
# ---------------------------------------------------------------------------


def save_url_to_local_work_dir(url: str, save_dir: str, save_filename: str = '') -> str:
    if not save_filename:
        save_filename = get_basename_from_url(url)
    new_path = os.path.join(save_dir, save_filename)
    if os.path.exists(new_path):
        os.remove(new_path)
    logger.info(f'Downloading {url} to {new_path}...')
    start_time = time.time()
    if not is_http_url(url):
        url = sanitize_chrome_file_path(url)
        shutil.copy(url, new_path)
    else:
        headers = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(new_path, 'wb') as file:
                file.write(response.content)
        else:
            raise ValueError('Can not download this file. Please check your network or the file link.')
    end_time = time.time()
    logger.info(f'Finished downloading {url} to {new_path}. Time spent: {end_time - start_time} seconds.')
    return new_path


# ---------------------------------------------------------------------------
# Read / write helpers
# ---------------------------------------------------------------------------


def save_text_to_file(path: str, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(text)


def read_text_from_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except UnicodeDecodeError:
        print_traceback(is_error=False)
        from charset_normalizer import from_path
        results = from_path(path)
        file_content = str(results.best())
    return file_content


# ---------------------------------------------------------------------------
# Content-type / file-type detection
# ---------------------------------------------------------------------------


def contains_html_tags(text: str) -> bool:
    pattern = r'<(p|span|div|li|html|script)[^>]*?'
    return bool(re.search(pattern, text))


def get_content_type_by_head_request(path: str) -> str:
    try:
        response = requests.head(path, timeout=5)
        return response.headers.get('Content-Type', '')
    except requests.RequestException:
        return 'unk'


def get_file_type(path: str) -> Literal['pdf', 'docx', 'pptx', 'txt', 'html', 'csv', 'tsv', 'xlsx', 'xls', 'unk']:
    f_type = get_basename_from_url(path).split('.')[-1].lower()
    if f_type in ('pdf', 'docx', 'pptx', 'csv', 'tsv', 'xlsx', 'xls'):
        return f_type

    if is_http_url(path):
        content_type = get_content_type_by_head_request(path)
        if 'application/pdf' in content_type:
            return 'pdf'
        elif 'application/msword' in content_type:
            return 'docx'
        return 'html'
    else:
        try:
            content = read_text_from_file(path)
        except Exception:
            print_traceback()
            return 'unk'

        if contains_html_tags(content):
            return 'html'
        else:
            return 'txt'
