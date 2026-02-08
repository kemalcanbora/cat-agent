"""llm.base package -- backward-compatible re-exports.

``from cat_agent.llm.base import BaseChatModel, register_llm, ...`` keeps working.
"""

from cat_agent.llm.base.model import (  # noqa: F401
    LLM_REGISTRY,
    BaseChatModel,
    ModelServiceError,
    register_llm,
)
from cat_agent.llm.base.retry import (  # noqa: F401
    retry_model_service,
    retry_model_service_iterator,
)
from cat_agent.llm.base.truncation import truncate_input_messages_roughly  # noqa: F401
