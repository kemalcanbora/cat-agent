"""Tool specification for synthesis."""

from __future__ import annotations

import builtins
import json
import keyword
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from cat_agent.log import logger
from cat_agent.tools.base import OPTIONAL_TOOL_REGISTRY, TOOL_REGISTRY

_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_HOLDOUT_SEED = 0xCA7A6E47  # fixed seed for reproducible splits


@dataclass
class Example:
    inputs: Dict[str, Any]
    expected: Any
    note: str = ''


@dataclass
class ParameterSpec:
    type: str
    description: str = ''


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, ParameterSpec]
    returns: str
    examples: List[Example]
    deps: List[str] = field(default_factory=list)
    holdout_ratio: float = 0.3
    requires_network: bool = False

    def __post_init__(self) -> None:
        # Normalise legacy Dict[str, str] forms passed positionally/by kwargs.
        normalised: Dict[str, ParameterSpec] = {}
        for key, value in dict(self.parameters or {}).items():
            normalised[str(key)] = coerce_parameter_spec(value)
        self.parameters = normalised
        validate_tool_spec(self)

    @property
    def registered_name(self) -> str:
        if self.name.startswith('generated_'):
            return self.name
        return f'generated_{self.name}'

    @property
    def function_name(self) -> str:
        if self.name.startswith('generated_'):
            return self.name[len('generated_'):]
        return self.name

    def split_examples(self) -> Tuple[List[Example], List[Example]]:
        """Return ``(work, holdout)`` with a fixed seed.

        Always keeps at least one work and one holdout example.
        """
        examples = list(self.examples)
        if len(examples) < 2:
            raise ValueError('Need at least 2 examples to form a holdout split.')
        rng = random.Random(_HOLDOUT_SEED)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        n_holdout = int(round(len(shuffled) * self.holdout_ratio))
        n_holdout = max(1, min(len(shuffled) - 1, n_holdout))
        holdout = shuffled[:n_holdout]
        work = shuffled[n_holdout:]
        return work, holdout

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Prefer the rich parameter form in artifacts.
        data['parameters'] = {
            name: {'type': param.type, 'description': param.description}
            for name, param in self.parameters.items()
        }
        return data


def coerce_parameter_spec(value: Any) -> ParameterSpec:
    if isinstance(value, ParameterSpec):
        return value
    if isinstance(value, str):
        return ParameterSpec(type=value, description='')
    if isinstance(value, dict):
        return ParameterSpec(
            type=str(value.get('type') or value.get('ty') or 'Any'),
            description=str(value.get('description') or ''),
        )
    raise ValueError(f'Invalid parameter spec: {value!r}')


def validate_tool_spec(spec: ToolSpec) -> None:
    name = spec.name
    if not name or not _NAME_RE.match(name):
        raise ValueError(
            f'Tool name {name!r} is not a valid Python identifier. '
            'Use letters, digits, and underscores; must not start with a digit.'
        )
    if keyword.iskeyword(name) or name in dir(builtins):
        raise ValueError(
            f'Tool name {name!r} shadows a Python builtin/keyword. Choose another name.'
        )

    registered = name if name.startswith('generated_') else f'generated_{name}'
    for candidate in {name, registered}:
        if candidate in TOOL_REGISTRY or candidate in OPTIONAL_TOOL_REGISTRY:
            raise ValueError(
                f'Tool name {candidate!r} is already registered. '
                'Choose a different name or unload the existing tool.'
            )

    if not spec.description or not str(spec.description).strip():
        raise ValueError('description must be a non-empty string.')

    if not spec.parameters:
        raise ValueError('parameters must list at least one parameter name → type.')

    for pname, pspec in spec.parameters.items():
        if not _NAME_RE.match(pname):
            raise ValueError(f'Parameter name {pname!r} is not a valid identifier.')
        if not pspec.type or not str(pspec.type).strip():
            raise ValueError(f'Parameter {pname!r} needs a type string.')
        if not (pspec.description or '').strip():
            logger.warning(
                'ToolSpec {!r}: parameter {!r} has no description; '
                'prefer {"type": "...", "description": "..."} so the model sees it.',
                name,
                pname,
            )

    if not spec.returns or not str(spec.returns).strip():
        raise ValueError('returns must describe the return type.')

    if len(spec.examples) < 3:
        raise ValueError(
            'Provide at least 3 input/output pairs, ideally including one edge case.'
        )

    for index, example in enumerate(spec.examples):
        if not isinstance(example.inputs, dict):
            raise ValueError(f'Example {index} inputs must be a dict.')
        for key in spec.parameters:
            if key not in example.inputs:
                raise ValueError(
                    f'Example {index} is missing required parameter {key!r}.'
                )

    if not (0.0 < spec.holdout_ratio < 1.0):
        raise ValueError('holdout_ratio must be between 0 and 1 (exclusive).')


def example_from_dict(data: Dict[str, Any]) -> Example:
    return Example(
        inputs=dict(data.get('inputs') or {}),
        expected=data.get('expected'),
        note=str(data.get('note') or ''),
    )


def tool_spec_from_dict(data: Dict[str, Any]) -> ToolSpec:
    examples_raw = data.get('examples') or []
    examples = [example_from_dict(item) for item in examples_raw]
    parameters = {
        str(k): coerce_parameter_spec(v)
        for k, v in (data.get('parameters') or {}).items()
    }
    return ToolSpec(
        name=str(data['name']),
        description=str(data.get('description') or ''),
        parameters=parameters,
        returns=str(data.get('returns') or 'Any'),
        examples=examples,
        deps=list(data.get('deps') or []),
        holdout_ratio=float(data.get('holdout_ratio', 0.3)),
        requires_network=bool(data.get('requires_network', False)),
    )


def load_tool_spec(path: Union[str, Path]) -> ToolSpec:
    """Load a :class:`ToolSpec` from a JSON or YAML file."""
    file_path = Path(path)
    text = file_path.read_text(encoding='utf-8')
    suffix = file_path.suffix.lower()
    if suffix in {'.yaml', '.yml'}:
        data = _load_yaml(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f'Spec file {file_path} must contain a JSON/YAML object.')
    return tool_spec_from_dict(data)


def _load_yaml(text: str) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            'PyYAML is required to load YAML tool specs. '
            'Install with: python3.10 -m pip install "cat-agent[synthesis]" '
            'or: python3.10 -m pip install pyyaml'
        ) from exc
    return yaml.safe_load(text)
