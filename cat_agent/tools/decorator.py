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

"""Function-based tool decorator that derives OpenAI-compatible schemas."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from docstring_parser import parse as parse_docstring
from pydantic import BaseModel

from cat_agent.tools.base import BaseTool, register_tool
from cat_agent.utils.json_utils import json_loads


class ToolArgumentError(ValueError):
    """Raised when a decorated tool receives invalid or missing arguments."""

    def __init__(self, tool_name: str, param_name: str, message: str):
        self.tool_name = tool_name
        self.param_name = param_name
        super().__init__(f"Tool `{tool_name}` parameter `{param_name}`: {message}")


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    allow_overwrite: bool = False,
    requires_network: bool = False,
    cloud_service: bool = False,
    register_by_default: bool = True,
) -> Any:
    """Register a plain function as a cat-agent tool.

    Supports both ``@tool`` and ``@tool(name='...')``. The decorated object is a
    :class:`BaseTool` instance that remains directly callable as the original
    function (including ``async`` callables).
    """

    def decorator(fn: Callable) -> BaseTool:
        tool_name = name or fn.__name__
        description, arg_docs = _parse_docstring(fn)
        type_hints = get_type_hints(fn)
        parameters = _build_parameters_schema(fn, type_hints, arg_docs)
        is_async = inspect.iscoroutinefunction(fn)

        def call(self, params: Union[str, dict], **kwargs) -> Any:
            args = _parse_and_coerce(tool_name, params, fn, type_hints)
            result = fn(**args)
            if inspect.isawaitable(result):
                return _run_coroutine(result)
            return result

        async def acall(self, params: Union[str, dict], **kwargs) -> Any:
            args = _parse_and_coerce(tool_name, params, fn, type_hints)
            result = fn(**args)
            if inspect.isawaitable(result):
                return await result
            return result

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            coerced: Dict[str, Any] = {}
            for key, value in bound.arguments.items():
                param = sig.parameters[key]
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    coerced[key] = value
                    continue
                hint = type_hints.get(key, inspect.Parameter.empty)
                if hint is inspect.Parameter.empty:
                    coerced[key] = value
                    continue
                try:
                    coerced[key] = _coerce_value(value, hint)
                except (TypeError, ValueError):
                    coerced[key] = value
            # Rebuild call respecting *args/**kwargs parameter kinds.
            call_args: list[Any] = []
            call_kwargs: Dict[str, Any] = {}
            for key, value in coerced.items():
                kind = sig.parameters[key].kind
                if kind is inspect.Parameter.VAR_POSITIONAL:
                    call_args.extend(value)
                elif kind is inspect.Parameter.VAR_KEYWORD:
                    call_kwargs.update(value)
                elif kind is inspect.Parameter.POSITIONAL_ONLY:
                    call_args.append(value)
                else:
                    call_kwargs[key] = value
            return fn(*call_args, **call_kwargs)

        attrs: Dict[str, Any] = {
            'name': tool_name,
            'description': description,
            'parameters': parameters,
            'call': call,
            '__call__': __call__,
            '_wrapped_func': staticmethod(fn),
            '_is_async': is_async,
        }
        # Async functions get a real coroutine acall; sync functions keep BaseTool.acall
        # (to_thread around call) so the agent never nests asyncio.run inside a worker.
        if is_async:
            attrs['acall'] = acall

        FunctionTool = type(
            f'{_to_camel(tool_name)}Tool',
            (BaseTool,),
            attrs,
        )

        register_tool(
            tool_name,
            allow_overwrite=allow_overwrite,
            requires_network=requires_network,
            cloud_service=cloud_service,
            register_by_default=register_by_default,
        )(FunctionTool)

        instance = FunctionTool()
        # Preserve useful function metadata on the instance for introspection.
        instance.__name__ = fn.__name__  # type: ignore[attr-defined]
        instance.__doc__ = fn.__doc__
        instance.__wrapped__ = fn  # type: ignore[attr-defined]
        instance.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return instance

    if func is not None:
        if not callable(func):
            raise TypeError('@tool expects a callable function')
        return decorator(func)
    return decorator


def _to_camel(snake: str) -> str:
    return ''.join(part.capitalize() or '_' for part in snake.split('_'))


def _parse_docstring(fn: Callable) -> tuple[str, Dict[str, str]]:
    doc = inspect.getdoc(fn) or ''
    parsed = parse_docstring(doc)
    description = (parsed.short_description or '').strip()
    if parsed.long_description:
        # Keep only the first paragraph as the tool description (short already
        # is the first paragraph for Google/NumPy/reST via docstring_parser).
        pass
    if not description and doc:
        description = doc.strip().split('\n\n', 1)[0].strip()
    arg_docs: Dict[str, str] = {}
    for param in parsed.params:
        if param.arg_name and param.description:
            arg_docs[param.arg_name] = param.description.strip()
    return description, arg_docs


def _build_parameters_schema(
    fn: Callable,
    type_hints: Dict[str, Any],
    arg_docs: Dict[str, str],
) -> dict:
    sig = inspect.signature(fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and p.name not in ('self', 'cls')
    ]

    # Single Pydantic BaseModel parameter → use its JSON schema directly.
    if len(params) == 1:
        hint = type_hints.get(params[0].name, params[0].annotation)
        model_cls = _as_basemodel(hint)
        if model_cls is not None:
            return _basemodel_to_parameters(model_cls)

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for param in params:
        if param.name not in type_hints and param.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"Tool `{fn.__name__}` parameter `{param.name}` is missing a type annotation"
            )
        hint = type_hints.get(param.name, param.annotation)
        prop = _annotation_to_json_schema(hint)
        if param.name in arg_docs:
            prop['description'] = arg_docs[param.name]
        properties[param.name] = prop
        if param.default is inspect.Parameter.empty and not _is_optional(hint):
            required.append(param.name)

    return {
        'type': 'object',
        'properties': properties,
        'required': required,
    }


def _as_basemodel(annotation: Any) -> Optional[type]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _basemodel_to_parameters(model_cls: type[BaseModel]) -> dict:
    schema = model_cls.model_json_schema()
    # Pydantic may nest $defs; keep a flat OpenAI-compatible object schema.
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    # Resolve local $ref pointers in properties when possible.
    defs = schema.get('$defs') or schema.get('definitions') or {}
    properties = _resolve_refs(properties, defs)
    return {
        'type': 'object',
        'properties': properties,
        'required': list(required),
    }


def _resolve_refs(obj: Any, defs: dict) -> Any:
    if isinstance(obj, dict):
        if '$ref' in obj and len(obj) == 1:
            ref = obj['$ref']
            # e.g. "#/$defs/Foo"
            name = ref.rsplit('/', 1)[-1]
            if name in defs:
                return _resolve_refs(defs[name], defs)
            return obj
        return {k: _resolve_refs(v, defs) for k, v in obj.items() if k != 'title'}
    if isinstance(obj, list):
        return [_resolve_refs(v, defs) for v in obj]
    return obj


def _is_optional(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return len(get_args(annotation)) == len(args) + 1
    return False


def _unwrap_optional(annotation: Any) -> Any:
    if _is_optional(annotation):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        return non_none[0] if len(non_none) == 1 else Union[tuple(non_none)]
    return annotation


def _annotation_to_json_schema(annotation: Any) -> dict:
    annotation = _unwrap_optional(annotation)

    model_cls = _as_basemodel(annotation)
    if model_cls is not None:
        schema = model_cls.model_json_schema()
        defs = schema.get('$defs') or schema.get('definitions') or {}
        return {
            'type': 'object',
            'properties': _resolve_refs(schema.get('properties', {}), defs),
            'required': list(schema.get('required', [])),
        }

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Literal[...] → string with enum (OpenAI-compatible tool schema convention)
    try:
        from typing import Literal
    except ImportError:  # pragma: no cover
        Literal = None  # type: ignore
    if Literal is not None and origin is Literal:
        values = list(args)
        return {
            'type': 'string',
            'enum': [v if isinstance(v, str) else str(v) for v in values],
        }

    if annotation is int:
        return {'type': 'integer'}
    if annotation is float:
        return {'type': 'number'}
    if annotation is str:
        return {'type': 'string'}
    if annotation is bool:
        return {'type': 'boolean'}
    if annotation is dict or origin in (dict, Dict):
        return {'type': 'object'}
    if annotation is list or origin in (list, List):
        item_schema = {'type': 'string'}
        if args:
            item_schema = _annotation_to_json_schema(args[0])
        return {'type': 'array', 'items': item_schema}

    # Fallback: treat unknown annotations as string.
    if annotation is Any or annotation is inspect.Parameter.empty:
        return {'type': 'string'}
    return {'type': 'string'}


def _parse_and_coerce(
    tool_name: str,
    params: Union[str, dict],
    fn: Callable,
    type_hints: Dict[str, Any],
) -> Dict[str, Any]:
    if isinstance(params, str):
        try:
            params_json: dict = json_loads(params) if params.strip() else {}
        except (json.JSONDecodeError, ValueError) as exc:
            raise ToolArgumentError(tool_name, '<params>', f'invalid JSON: {exc}') from exc
    elif isinstance(params, dict):
        params_json = dict(params)
    else:
        raise ToolArgumentError(tool_name, '<params>', f'expected str or dict, got {type(params).__name__}')

    sig = inspect.signature(fn)
    params_list = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and p.name not in ('self', 'cls')
    ]

    # Mirror schema derivation: a lone BaseModel param uses the model schema at
    # the top level, so the whole JSON object is that model (not nested under the
    # parameter name).
    if len(params_list) == 1:
        only = params_list[0]
        hint = type_hints.get(only.name, only.annotation)
        if _as_basemodel(hint) is not None:
            # Prefer nested form if the model was passed under its param name.
            raw = params_json[only.name] if only.name in params_json else params_json
            try:
                return {only.name: _coerce_value(raw, hint)}
            except (TypeError, ValueError) as exc:
                raise ToolArgumentError(
                    tool_name,
                    only.name,
                    f'failed to coerce value {raw!r} to {hint}: {exc}',
                ) from exc

    result: Dict[str, Any] = {}
    for param in params_list:
        hint = type_hints.get(param.name, param.annotation)
        has_default = param.default is not inspect.Parameter.empty
        is_opt = _is_optional(hint)

        if param.name not in params_json:
            if has_default or is_opt:
                if has_default:
                    result[param.name] = param.default
                else:
                    result[param.name] = None
                continue
            raise ToolArgumentError(tool_name, param.name, 'required argument is missing')

        raw = params_json[param.name]
        try:
            result[param.name] = _coerce_value(raw, hint)
        except (TypeError, ValueError) as exc:
            raise ToolArgumentError(
                tool_name,
                param.name,
                f'failed to coerce value {raw!r} to {hint}: {exc}',
            ) from exc

    return result


def _coerce_value(value: Any, annotation: Any) -> Any:
    annotation = _unwrap_optional(annotation)
    if value is None:
        return None

    model_cls = _as_basemodel(annotation)
    if model_cls is not None:
        if isinstance(value, model_cls):
            return value
        if isinstance(value, dict):
            return model_cls.model_validate(value)
        raise TypeError(f'expected object for {model_cls.__name__}')

    origin = get_origin(annotation)
    args = get_args(annotation)

    try:
        from typing import Literal
    except ImportError:  # pragma: no cover
        Literal = None  # type: ignore
    if Literal is not None and origin is Literal:
        if value in args:
            return value
        # Coerce string forms of literal members when possible.
        for lit in args:
            if str(value) == str(lit):
                return lit
        raise ValueError(f'value must be one of {list(args)}')

    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ('true', '1', 'yes'):
                return True
            if lowered in ('false', '0', 'no'):
                return False
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        raise TypeError('expected boolean')

    if annotation is int:
        if isinstance(value, bool):
            raise TypeError('expected integer')
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            return int(value.strip())
        return int(value)

    if annotation is float:
        if isinstance(value, bool):
            raise TypeError('expected number')
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        return float(value)

    if annotation is str:
        return value if isinstance(value, str) else str(value)

    if annotation is list or origin in (list, List):
        if not isinstance(value, list):
            raise TypeError('expected array')
        if args:
            return [_coerce_value(item, args[0]) for item in value]
        return value

    if annotation is dict or origin in (dict, Dict):
        if not isinstance(value, dict):
            raise TypeError('expected object')
        return value

    return value


def _run_coroutine(coro: Any) -> Any:
    """Run a coroutine from sync ``BaseTool.call`` (same idea as MCPManager)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside an event loop: schedule and wait via a new loop in a thread
    # would risk deadlock; raise a clear error instead.
    if loop.is_running():
        raise RuntimeError(
            'Cannot execute async tool via .call() from a running event loop; '
            'await the decorated function directly instead.'
        )
    return loop.run_until_complete(coro)
