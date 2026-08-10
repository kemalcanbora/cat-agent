#!/usr/bin/env python3.10
"""Minimal smoke test: synthesise, load, register, and call a generated tool.

Config from repo ``.env`` (no shell export needed)::

    CAT_AGENT_OFFLINE=0
    OLLAMA_API_KEY=...
    LLM_MODEL=minimax-m2.5:cloud
    OLLAMA_API_BASE=https://ollama.com/v1

    python3.10 examples/synthesis/from_spec/run_synthesis.py
    python3.10 examples/synthesis/from_spec/run_synthesis.py --spec examples/synthesis/from_spec/kdv_spec.json

Ollama has no dedicated backend — uses OpenAI-compatible ``model_type=oai``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

# Must load BEFORE importing cat_agent (offline guards read CAT_AGENT_OFFLINE at import).
load_dotenv(REPO_ROOT / '.env', override=True)

from cat_agent.synthesis import (  # noqa: E402
    Status,
    ToolSmith,
    get_executor,
    load_generated_tools,
    load_tool_spec,
)
from cat_agent.tools.base import enable_optional_tools  # noqa: E402


def build_llm_cfg() -> Dict:
    """Ollama Cloud / local via the existing OpenAI-compatible ``oai`` backend."""
    api_key = (
        os.getenv('OLLAMA_API_KEY')
        or os.getenv('OPENAI_API_KEY')
        or 'EMPTY'
    )
    model = os.getenv('LLM_MODEL', 'minimax-m2.7:cloud')
    base_url = (os.getenv('OLLAMA_API_BASE') or 'https://ollama.com/v1').rstrip('/')
    if not base_url.endswith('/v1'):
        base_url = base_url + '/v1'
    return {
        'model': model,
        'model_type': 'oai',
        'model_server': base_url,
        'api_key': api_key,
        'generate_cfg': {
            'temperature': 0.2,
            'top_p': 0.8,
            'max_tokens': 8192,
        },
    }


def main() -> None:
    if not (os.getenv('OLLAMA_API_KEY') or os.getenv('OPENAI_API_KEY')):
        print(
            f'Missing OLLAMA_API_KEY in {REPO_ROOT / ".env"} — see .env.example.'
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Synthesise a sandboxed tool from a spec.')
    parser.add_argument(
        '--spec',
        default=str(
            REPO_ROOT / 'examples' / 'synthesis' / 'from_spec' / 'add_one_spec.json'
        ),
        help='Path to a ToolSpec JSON (or YAML) file',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Workspace root for generated_tools/ (default: cat-agent workspace)',
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.is_file():
        print(f'Spec not found: {spec_path}')
        sys.exit(1)

    llm_cfg = build_llm_cfg()
    print(f'LLM: model={llm_cfg["model"]} server={llm_cfg["model_server"]}')
    print(f'Spec: {spec_path}')

    try:
        executor = get_executor('wasm')
    except Exception as exc:
        print(f'WASM executor unavailable: {exc}')
        print('Install with: python3.10 -m pip install "cat-agent[wasm]"')
        sys.exit(1)

    smith = ToolSmith(
        llm=llm_cfg,
        executor=executor,
        max_attempts=5,
        output_dir=args.output_dir,
    )
    spec = load_tool_spec(spec_path)
    result = smith.synthesize(spec)

    print(f'status={result.status.value} ok={result.ok}')
    if result.error:
        print(f'error: {result.error}')
    if result.status == Status.HOLDOUT_FAILED:
        print('Holdout failures (add these to your spec and re-run):')
        for case in result.holdout_failures:
            print(
                f'  inputs={case.get("inputs")!r} '
                f'expected={case.get("expected")!r} '
                f'actual={case.get("returned")!r}'
            )
        sys.exit(2)

    if not result.ok:
        sys.exit(1)

    print(f'artifacts: {result.artifact_dir}')
    print(f'registered_name: {result.registered_name}')

    # Opt-in load + enable, then call once with the first example.
    artifact_root = Path(result.artifact_dir).parent
    loaded = load_generated_tools(path=str(artifact_root))
    enable_optional_tools(result.registered_name)
    tool = loaded[result.registered_name]
    sample = spec.examples[0].inputs
    print(f'call({sample!r}) → {tool(**sample)!r}')


if __name__ == '__main__':
    main()
