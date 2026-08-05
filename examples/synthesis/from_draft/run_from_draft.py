#!/usr/bin/env python3.10
"""Interactive intake: Markdown draft → interview → sandboxed tool.

Config from repo ``.env``::

    CAT_AGENT_OFFLINE=0
    OLLAMA_API_KEY=...
    LLM_MODEL=minimax-m2.5:cloud
    OLLAMA_BASE_URL=https://ollama.com/v1
    # optional stronger model for intake:
    # INTAKE_LLM_MODEL=...

    python3.10 examples/synthesis/from_draft/run_from_draft.py
    python3.10 examples/synthesis/from_draft/run_from_draft.py --draft examples/synthesis/from_draft/vat_draft_de.md
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

load_dotenv(REPO_ROOT / '.env', override=True)

from cat_agent.synthesis import synthesize_from_draft  # noqa: E402
from cat_agent.synthesis.intake.interview import Question  # noqa: E402


def build_llm_cfg(model: str | None = None) -> Dict:
    api_key = (
        os.getenv('OLLAMA_API_KEY')
        or os.getenv('OPENAI_API_KEY')
        or 'EMPTY'
    )
    chosen = model or os.getenv('LLM_MODEL', 'minimax-m2.7:cloud')
    base_url = (os.getenv('OLLAMA_BASE_URL') or 'https://ollama.com/v1').rstrip('/')
    if not base_url.endswith('/v1'):
        base_url = base_url + '/v1'
    return {
        'model': chosen,
        'model_type': 'oai',
        'model_server': base_url,
        'api_key': api_key,
        'generate_cfg': {
            'temperature': 0.2,
            'top_p': 0.8,
            # Reasoning models (Nemotron etc.) spend tokens on thinking first;
            # 1024 often yields empty content and "No code produced".
            'max_tokens': 8192,
        },
    }


def ask(question: Question) -> str:
    print()
    print(question.text)
    print(f'[{question.pending_sentinel}]')
    return input('> ').strip()


def main() -> None:
    if not (os.getenv('OLLAMA_API_KEY') or os.getenv('OPENAI_API_KEY')):
        print(
            f'Missing OLLAMA_API_KEY in {REPO_ROOT / ".env"} — see .env.example.'
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Synthesise a tool from a Markdown draft.')
    parser.add_argument(
        '--draft',
        default=str(
            REPO_ROOT / 'examples' / 'synthesis' / 'from_draft' / 'vat_draft.md'
        ),
    )
    parser.add_argument('--locale', default=None)
    parser.add_argument('--lang', default=None)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    llm_cfg = build_llm_cfg()
    intake_model = os.getenv('INTAKE_LLM_MODEL')
    intake_cfg = build_llm_cfg(intake_model) if intake_model else llm_cfg
    print(f'LLM (synthesis): {llm_cfg["model"]}')
    print(f'LLM (intake):     {intake_cfg["model"]}')
    print(f'Draft: {args.draft}')

    result = synthesize_from_draft(
        args.draft,
        llm=llm_cfg,
        intake_llm=intake_cfg,
        locale=args.locale,
        lang=args.lang,
        ask=ask,
        output_dir=args.output_dir,
    )
    if result.ok and result.synthesis:
        print(f'success → {result.synthesis.artifact_dir}')
        sys.exit(0)
    print(f'failed: {result.error}')
    sys.exit(1)


if __name__ == '__main__':
    main()
