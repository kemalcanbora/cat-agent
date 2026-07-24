import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from cat_agent.llm.schema import ContentItem
from cat_agent.settings import DEFAULT_WORKSPACE
from cat_agent.utils.utils import (
    has_chinese_chars,
    json_loads,
    logger,
    print_traceback,
    save_url_to_local_work_dir,
)

TOOL_REGISTRY = {}
OPTIONAL_TOOL_REGISTRY = {}


class ToolServiceError(Exception):

    def __init__(
        self,
        exception: Optional[Exception] = None,
        code: Optional[str] = None,
        message: Optional[str] = None,
        extra: Optional[dict] = None,
    ):
        if exception is not None:
            super().__init__(exception)
        else:
            super().__init__(f"\nError code: {code}. Error message: {message}")
        self.exception = exception
        self.code = code
        self.message = message
        self.extra = extra


class ToolNotFoundError(ToolServiceError):
    def __init__(self, tool_name: str):
        message = f'Tool {tool_name} does not exist.'
        super().__init__(message=message)
        self.tool_name = tool_name


class ToolExecutionError(ToolServiceError):
    def __init__(self, tool_name: str, message: str):
        super().__init__(message=message)
        self.tool_name = tool_name


def register_tool(
    name,
    allow_overwrite=False,
    *,
    requires_network: bool = False,
    cloud_service: bool = False,
    register_by_default: bool = True,
):
    def decorator(cls):
        from cat_agent.security.tool_policy import record_tool_metadata

        cls.requires_network = requires_network
        cls.cloud_service = cloud_service
        record_tool_metadata(
            name,
            requires_network=requires_network,
            cloud_service=cloud_service,
            register_by_default=register_by_default,
        )
        target = TOOL_REGISTRY if register_by_default else OPTIONAL_TOOL_REGISTRY
        if name in target and not allow_overwrite:
            registry_label = 'Tool' if register_by_default else 'Optional tool'
            raise ValueError(f'{registry_label} `{name}` already exists! Please ensure that the tool name is unique.')
        if cls.name and (cls.name != name):
            raise ValueError(f'{cls.__name__}.name="{cls.name}" conflicts with @register_tool(name="{name}").')
        cls.name = name
        if allow_overwrite and name in target:
            logger.warning(f'Tool `{name}` already exists! Overwriting with class {cls}.')
        target[name] = cls
        if register_by_default:
            OPTIONAL_TOOL_REGISTRY.pop(name, None)
        return cls

    return decorator


def enable_optional_tools(*names: str) -> None:
    """Move opt-in network/cloud tools into the active registry."""
    if not names:
        names = tuple(OPTIONAL_TOOL_REGISTRY.keys())
    for name in names:
        if name not in OPTIONAL_TOOL_REGISTRY:
            continue
        TOOL_REGISTRY[name] = OPTIONAL_TOOL_REGISTRY.pop(name)


def is_tool_allowed_for_agent(tool_name: str, tool_cls: type) -> bool:
    from cat_agent.security.tool_policy import is_tool_allowed_in_offline_mode

    return is_tool_allowed_in_offline_mode(tool_name, tool_cls)


def is_tool_schema(obj: dict) -> bool:
    """
    Check if obj is a valid JSON schema describing a tool compatible with OpenAI's tool calling.
    Example valid schema:
    {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
    """
    import jsonschema

    try:
        assert set(obj.keys()) == {"name", "description", "parameters"}
        assert isinstance(obj["name"], str)
        assert obj["name"].strip()
        assert isinstance(obj["description"], str)
        assert isinstance(obj["parameters"], dict)

        assert set(obj["parameters"].keys()) == {"type", "properties", "required"}
        assert obj["parameters"]["type"] == "object"
        assert isinstance(obj["parameters"]["properties"], dict)
        assert isinstance(obj["parameters"]["required"], list)
        assert set(obj["parameters"]["required"]).issubset(set(obj["parameters"]["properties"].keys()))
    except AssertionError:
        return False
    try:
        jsonschema.validate(instance={}, schema=obj["parameters"])
    except jsonschema.exceptions.SchemaError:
        return False
    except jsonschema.exceptions.ValidationError:
        pass
    return True


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: Union[List[dict], dict, None] = None  # avoid mutable default
    requires_network: bool = False
    cloud_service: bool = False

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        if not self.name:
            raise ValueError(
                f"You must set {self.__class__.__name__}.name, either by @register_tool(name=...) or explicitly setting {self.__class__.__name__}.name"
            )

        # Normalize parameters per instance (no shared mutable defaults)
        if self.parameters is None:
            self.parameters = []
        elif isinstance(self.parameters, tuple):
            self.parameters = list(self.parameters)

        if isinstance(self.parameters, dict):
            if not is_tool_schema({"name": self.name, "description": self.description, "parameters": self.parameters}):
                raise ValueError(
                    "The parameters, when provided as a dict, must confirm to a valid openai-compatible JSON schema."
                )

    @abstractmethod
    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict, List[ContentItem]]:
        """The interface for calling tools.

        Each tool needs to implement this function, which is the workflow of the tool.

        Args:
            params: The parameters of func_call.
            kwargs: Additional parameters for calling tools.

        Returns:
            The result returned by the tool, implemented in the subclass.
        """
        raise NotImplementedError

    async def acall(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict, List[ContentItem]]:
        """Async interface for calling tools.

        Default implementation runs sync :meth:`call` in a worker thread so the
        event loop is not blocked. Subclasses with native coroutines (or the
        ``@tool`` decorator for ``async def`` functions) should override this.
        """
        return await asyncio.to_thread(self.call, params, **kwargs)

    def _verify_json_format_args(self, params: Union[str, dict], strict_json: bool = False) -> dict:
        """Verify the parameters of the function call"""
        if isinstance(params, str):
            try:
                if strict_json:
                    params_json: dict = json.loads(params)
                else:
                    params_json: dict = json_loads(params)
            except json.decoder.JSONDecodeError:
                raise ValueError("Parameters must be formatted as a valid JSON!")
        else:
            params_json: dict = params
        if isinstance(self.parameters, list):
            for param in self.parameters:
                if "required" in param and param["required"]:
                    if param["name"] not in params_json:
                        raise ValueError("Parameters %s is required!" % param["name"])
        elif isinstance(self.parameters, dict):
            import jsonschema

            jsonschema.validate(instance=params_json, schema=self.parameters)
        else:
            raise ValueError
        return params_json

    @property
    def function(self) -> dict:  # Bad naming. It should be `function_info`.
        return {
            # 'name_for_human': self.name_for_human,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            # 'args_format': self.args_format
        }

    @property
    def name_for_human(self) -> str:
        return self.cfg.get("name_for_human", self.name)

    @property
    def args_format(self) -> str:
        fmt = self.cfg.get("args_format")
        if fmt is None:
            if has_chinese_chars([self.name_for_human, self.name, self.description, self.parameters]):
                fmt = "The input for this tool should be a JSON object."
            else:
                fmt = "Format the arguments as a JSON object."
        return fmt

    @property
    def file_access(self) -> bool:
        return False


class BaseToolWithFileAccess(BaseTool, ABC):
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        assert self.name
        default_work_dir = os.path.join(DEFAULT_WORKSPACE, "tools", self.name)
        self.work_dir: str = self.cfg.get("work_dir", default_work_dir)

    @property
    def file_access(self) -> bool:
        return True

    @abstractmethod
    def call(self, params: Union[str, dict], files: List[str] = None, **kwargs) -> str:
        """
        Subclasses must implement this method.
        Implementations may optionally use `files`, which—if provided—will have been copied into `self.work_dir`.
        """
        if files:
            os.makedirs(self.work_dir, exist_ok=True)
            for file in files:
                try:
                    save_url_to_local_work_dir(file, self.work_dir)
                except Exception:
                    print_traceback()
        # Subclasses should return a string result.
        raise NotImplementedError