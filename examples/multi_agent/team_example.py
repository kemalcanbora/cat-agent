"""Three-agent Earth equatorial-speed demo (round-robin GroupChat).

  DataGuy → PhysicsGuy → Explainer  (each with its own model id)

Models — same gateway, different ids (see ``agent.yaml``)::

  CAT_AGENT_LLM_MODEL_DATAGUY / PHYSICSGUY / (Explainer = model.alias)

    python examples/multi_agent/team_example.py
    cat-agent serve --factory team_example:registry
    cat-agent deploy --dir examples/multi_agent
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from cat_agent.agents import Assistant, GroupChat
from cat_agent.llm.env_config import apply_agent_yaml_env, llm_config_from_env
from cat_agent.llm.schema import ASSISTANT, FUNCTION, Message
from cat_agent.multi_agent import Blackboard, HubEvent
from cat_agent.observability import PrintHandler
from cat_agent.serve import AgentRegistry
from cat_agent.tools import tool

_R = 6_378_137.0
_T = 86_164.0905
_YAML = Path(__file__).resolve().parent / 'agent.yaml'


@tool(allow_overwrite=True)
def earth_rotation_constants() -> str:
    """Earth equatorial radius R (m) and sidereal day T (s)."""
    return f'Earth equatorial radius R = {_R:g} m. Sidereal day T = {_T:g} s.'


@tool(allow_overwrite=True)
def equatorial_speed(radius_m: float, period_s: float) -> str:
    """Equatorial speed v = 2πR / T (m/s and km/h).

    Args:
        radius_m: Equatorial radius in meters
        period_s: Sidereal day in seconds
    """
    v = 2.0 * math.pi * radius_m / period_s
    return f'v = 2πR/T = {v:.2f} m/s ≈ {v * 3.6:.1f} km/h'


class EarthSpinTeam:
    """Build the three-agent GroupChat."""

    def llm(self, model: Optional[str] = None) -> Dict[str, Any]:
        cfg = llm_config_from_env(
            agent_yaml=_YAML if _YAML.is_file() else None,
            model=model,
            model_type='oai',
            generate_cfg={'temperature': 0.2, 'top_p': 0.8, 'max_tokens': 512},
        )
        if 'base_url' in cfg and 'model_server' not in cfg:
            cfg['model_server'] = cfg['base_url']
        return cfg

    @staticmethod
    def model_for(env_key: str, fallback: str) -> str:
        return (os.getenv(env_key) or '').strip() or fallback

    @staticmethod
    def _on_hub(event: HubEvent) -> None:
        if event.type in ('agent_start', 'agent_end', 'ask', 'handoff'):
            print(f'  [hub] {event.type:12} agent={event.agent}')

    def build(self) -> GroupChat:
        # Local: agent.yaml env (per-agent models) is not auto-loaded like Nomad.
        apply_agent_yaml_env(_YAML)
        explainer_cfg = self.llm()
        if not (explainer_cfg.get('base_url') or explainer_cfg.get('model_server')):
            raise RuntimeError(
                'Missing LLM base URL — set CAT_AGENT_LLM_BASE_URL or OPENAI_BASE_URL'
            )
        if not explainer_cfg.get('api_key'):
            raise RuntimeError(
                'Missing LLM API key — set OLLAMA_API_KEY or OPENAI_API_KEY '
                '(or CAT_AGENT_LLM_API_KEY) in the repo .env'
            )
        primary = str(explainer_cfg.get('model') or 'default')
        data_m = self.model_for('CAT_AGENT_LLM_MODEL_DATAGUY', primary)
        phys_m = self.model_for('CAT_AGENT_LLM_MODEL_PHYSICSGUY', primary)

        return GroupChat(
            agents=[
                Assistant(
                    llm=self.llm(data_m),
                    name='DataGuy',
                    description='Looks up Earth radius and sidereal day.',
                    system_message=(
                        'You are DataGuy. Call earth_rotation_constants, share R and T. '
                        'Optional: write_artifact key earth_constants.'
                    ),
                    function_list=['earth_rotation_constants'],
                ),
                Assistant(
                    llm=self.llm(phys_m),
                    name='PhysicsGuy',
                    description='Computes equatorial speed via equatorial_speed.',
                    system_message=(
                        'You are PhysicsGuy. Use DataGuy\'s R, T (or read_artifact '
                        'earth_constants), call equatorial_speed, reply with the result.'
                    ),
                    function_list=['equatorial_speed'],
                ),
                Assistant(
                    llm=explainer_cfg,
                    name='Explainer',
                    description='Explains the speed for beginners.',
                    system_message=(
                        'You are Explainer. In 2–3 sentences explain PhysicsGuy\'s '
                        'speed vs a jet (~900 km/h). Do not recompute.'
                    ),
                ),
            ],
            agent_selection_method='round_robin',
            on_event=self._on_hub,
            blackboard=Blackboard(),
            inject_hub_tools=True,
            name='EarthSpinTeam',
        )


def registry() -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(EarthSpinTeam().build(), name='earth-spin')
    return reg


def main() -> None:
    team = EarthSpinTeam()
    chat = team.build()
    cfg = team.llm()
    primary = str(cfg.get('model') or 'default')
    print(
        f'LLM {cfg.get("model_server") or cfg.get("base_url")}\n'
        f'  DataGuy    → {team.model_for("CAT_AGENT_LLM_MODEL_DATAGUY", primary)}\n'
        f'  PhysicsGuy → {team.model_for("CAT_AGENT_LLM_MODEL_PHYSICSGUY", primary)}\n'
        f'  Explainer  → {primary}'
    )

    prompt = (
        "How fast is a point on Earth's equator moving because of Earth's rotation? "
        "Use the sidereal day and Earth's equatorial radius, then explain for a beginner."
    )
    transcript: List[Message] = []
    for batch in chat.run(
        messages=[Message(role='user', content=prompt, name='user')],
        max_round=3,
        handlers=[PrintHandler()],
    ):
        transcript = batch

    print('=== conversation ===')
    print(f'user: {prompt}')
    for msg in transcript:
        if msg.role == ASSISTANT and msg.function_call:
            fc = msg.function_call
            print(f'\n{msg.name or "assistant"} → tool {fc.name}({fc.arguments})')
        elif msg.role == FUNCTION:
            print(f'  ↩ {msg.name}: {(msg.content or "")[:200]}')
        elif msg.role == ASSISTANT and msg.name and msg.content:
            print(f'\n{msg.name}:\n  {msg.content}')
    print('\nBlackboard:', chat.blackboard.describe())


if __name__ == '__main__':
    main()
