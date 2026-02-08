"""Retry helpers with exponential backoff for model-service calls."""

import random
import time
from typing import Any, Iterator, Tuple

from cat_agent.log import logger


def retry_model_service(
    fn,
    max_retries: int = 10,
) -> Any:
    """Retry a synchronous callable that may raise ``ModelServiceError``."""
    from cat_agent.llm.base.model import ModelServiceError  # deferred to avoid circular import

    num_retries, delay = 0, 1.0
    while True:
        try:
            return fn()
        except ModelServiceError as e:
            num_retries, delay = _raise_or_delay(e, num_retries, delay, max_retries)


def retry_model_service_iterator(
    it_fn,
    max_retries: int = 10,
) -> Iterator:
    """Retry an iterator-returning callable that may raise ``ModelServiceError``."""
    from cat_agent.llm.base.model import ModelServiceError

    num_retries, delay = 0, 1.0
    while True:
        try:
            for rsp in it_fn():
                yield rsp
            break
        except ModelServiceError as e:
            num_retries, delay = _raise_or_delay(e, num_retries, delay, max_retries)


def _raise_or_delay(
    e,
    num_retries: int,
    delay: float,
    max_retries: int = 10,
    max_delay: float = 300.0,
    exponential_base: float = 2.0,
) -> Tuple[int, float]:
    """Re-raise non-retryable errors; otherwise sleep with exponential backoff."""
    from cat_agent.llm.base.model import ModelServiceError

    if max_retries <= 0:
        raise e

    # Non-retryable conditions
    if e.code == '400':
        raise e
    if e.code == 'DataInspectionFailed':
        raise e
    if 'inappropriate content' in str(e):
        raise e
    if 'maximum context length' in str(e):
        raise e

    logger.warning('ModelServiceError - ' + str(e).strip('\n'))

    if num_retries >= max_retries:
        raise ModelServiceError(exception=Exception(f'Maximum number of retries ({max_retries}) exceeded.'))

    num_retries += 1
    jitter = 1.0 + random.random()
    delay = min(delay * exponential_base, max_delay) * jitter
    time.sleep(delay)
    return num_retries, delay
