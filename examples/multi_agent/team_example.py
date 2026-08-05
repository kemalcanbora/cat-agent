"""Three-agent collaboration: Earth's equatorial rotation speed.

Round-robin GroupChat. Each specialist is a normal ``Assistant`` with tools in
``function_list`` — the LLM decides when to call them (no manual ``tool.call``).

  1. DataGuy     — ``earth_rotation_constants``
  2. PhysicsGuy  — ``equatorial_speed`` using DataGuy's R, T
  3. Explainer   — explains what the others said

Config from repo ``.env`` (no shell export needed)::

    CAT_AGENT_OFFLINE=0
    OLLAMA_API_KEY=...
    LLM_MODEL=minimax-m2.5:cloud
    OLLAMA_BASE_URL=https://ollama.com/v1

    python examples/multi_agent/math_team_example.py

Ollama has no dedicated backend — uses OpenAI-compatible ``model_type=oai``.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

# Must load BEFORE importing cat_agent (offline guards read CAT_AGENT_OFFLINE at import).
# override=True so repo .env wins over PyCharm Run Configuration env.
load_dotenv(REPO_ROOT / '.env', override=True)

from cat_agent.agents import Assistant, GroupChat
from cat_agent.llm.schema import ASSISTANT, FUNCTION, Message
from cat_agent.multi_agent import Blackboard, HubEvent
from cat_agent.observability import PrintHandler
from cat_agent.tools import tool

EARTH_RADIUS_M = 6_378_137.0
SIDEREAL_DAY_S = 86_164.0905


@tool(allow_overwrite=True)
def earth_rotation_constants() -> str:
    """Return reference constants for Earth rotation: equatorial radius R in
    meters and sidereal day length T in seconds.
    """
    return (
        f'Earth equatorial radius R = {EARTH_RADIUS_M:g} m. '
        f'Sidereal day T = {SIDEREAL_DAY_S:g} s '
        f'(one full rotation relative to the stars).'
    )


@tool(allow_overwrite=True)
def equatorial_speed(radius_m: float, period_s: float) -> str:
    """Compute equatorial linear speed v = 2πR / T.
    Pass radius_m and period_s from DataGuy (or artifact:earth_constants).
    Returns m/s and km/h.

    Args:
        radius_m: Equatorial radius in meters
        period_s: Sidereal day in seconds
    """
    v = 2.0 * math.pi * radius_m / period_s
    return (
        f'v = 2πR / T = 2π·{radius_m:g} / {period_s:g} '
        f'= {v:.2f} m/s ≈ {v * 3.6:.1f} km/h'
    )


def build_llm_cfg() -> Dict:
    """Ollama Cloud / local via the existing OpenAI-compatible ``oai`` backend."""
    api_key = (
        os.getenv('OLLAMA_API_KEY')
        or os.getenv('OPENAI_API_KEY')
        or 'EMPTY'
    )
    model = os.getenv('LLM_MODEL', 'minimax-m2.7:cloud')
    base_url = (os.getenv('OLLAMA_BASE_URL') or 'https://ollama.com/v1').rstrip('/')
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
            'max_tokens': 512,
        },
    }


def on_hub_event(event: HubEvent) -> None:
    if event.type in ('agent_start', 'agent_end', 'ask', 'handoff'):
        print(f'  [hub] {event.type:12} agent={event.agent}')


def build_team() -> GroupChat:
    llm_cfg = build_llm_cfg()
    print(f'LLM: model={llm_cfg["model"]} server={llm_cfg["model_server"]}')
    print(f'OFFLINE={os.getenv("CAT_AGENT_OFFLINE", "")!r}')

    data_guy = Assistant(
        llm=llm_cfg,
        name='DataGuy',
        description='Looks up Earth radius and sidereal day via earth_rotation_constants.',
        system_message=(
            'You are DataGuy. Your job is to provide Earth rotation constants. '
            'Use the earth_rotation_constants tool, then share R and T briefly. '
            'You may store the tool result with write_artifact key earth_constants.'
        ),
        function_list=['earth_rotation_constants'],
    )

    physics_guy = Assistant(
        llm=llm_cfg,
        name='PhysicsGuy',
        description='Computes equatorial speed v=2πR/T via the equatorial_speed tool.',
        system_message=(
            'You are PhysicsGuy. Read R and T from DataGuy (chat or '
            'read_artifact earth_constants). Then call equatorial_speed with those '
            'values. Reply with the tool result. Optionally write_artifact '
            'equatorial_speed. Do not invent R or T yourself.'
        ),
        function_list=['equatorial_speed'],
    )

    explainer = Assistant(
        llm=llm_cfg,
        name='Explainer',
        description='Explains the computed Earth rotation speed for beginners.',
        system_message=(
            'You are Explainer. DataGuy and PhysicsGuy already spoke. '
            'In 2–3 short sentences, explain PhysicsGuy\'s speed for a beginner '
            'and compare it to a jet (~900 km/h). Do not recompute.'
        ),
    )

    return GroupChat(
        agents=[data_guy, physics_guy, explainer],
        agent_selection_method='round_robin',
        on_event=on_hub_event,
        blackboard=Blackboard(),
        inject_hub_tools=True,
        name='EarthSpinTeam',
    )


def _print_transcript(transcript: List[Message], prompt: str) -> None:
    print('=== conversation ===')
    print(f'user: {prompt}')
    for msg in transcript:
        if msg.role == ASSISTANT and msg.function_call:
            fc = msg.function_call
            print(f'\n{msg.name or "assistant"} → tool {fc.name}({fc.arguments})')
        elif msg.role == FUNCTION:
            preview = (msg.content or '')[:200]
            print(f'  ↩ {msg.name}: {preview}')
        elif msg.role == ASSISTANT and msg.name and msg.content:
            print(f'\n{msg.name}:\n  {msg.content}')


def main() -> None:
    if not (os.getenv('OLLAMA_API_KEY') or os.getenv('OPENAI_API_KEY')):
        print(
            f'Missing OLLAMA_API_KEY in {REPO_ROOT / ".env"} — see .env.example.'
        )
        sys.exit(1)

    team = build_team()

    prompt = (
        'How fast is a point on Earth\'s equator moving because of Earth\'s rotation? '
        'Use the sidereal day and Earth\'s equatorial radius, then explain for a beginner.'
    )
    messages = [Message(role='user', content=prompt, name='user')]

    v_expected = 2.0 * math.pi * EARTH_RADIUS_M / SIDEREAL_DAY_S

    print('=' * 60)
    print('Scenario (round_robin, LLM chooses tools):')
    print('  1. DataGuy     → earth_rotation_constants')
    print('  2. PhysicsGuy  → equatorial_speed (from DataGuy\'s R, T)')
    print('  3. Explainer   → beginner explanation')
    print(f'User: {prompt}')
    print(f'Expected ≈ {v_expected:.2f} m/s ({v_expected * 3.6:.1f} km/h)')
    print('Watch for "→ tool ..." lines — those are real function calls.')
    print('---')

    transcript: List[Message] = []
    for batch in team.run(
        messages=messages,
        max_round=3,
        handlers=[PrintHandler()],
    ):
        transcript = batch

    print()
    _print_transcript(transcript, prompt)
    print()
    print('Blackboard:', team.blackboard.describe())


if __name__ == '__main__':
    main()
