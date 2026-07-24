"""PhysicsGuy — async Assistant that can call multiple tools in one turn.

Uses local ``llama_cpp`` (same model as ``examples/llama_cpp_math_guy``). When the
model emits several tool calls together, ``arun`` runs them concurrently.

    PYTHONPATH=. python examples/async_agent/physics_guy.py
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cat_agent.agents import Assistant
from cat_agent.tools import tool


@tool(allow_overwrite=True)
async def kinetic_energy(mass_kg: float, velocity_m_s: float) -> str:
    """Compute kinetic energy KE = ½ m v².

    Args:
        mass_kg: Mass in kilograms
        velocity_m_s: Speed in meters per second
    """
    ke = 0.5 * mass_kg * velocity_m_s * velocity_m_s
    return f'KE = ½·{mass_kg:g}·({velocity_m_s:g})² = {ke:.4g} J'


@tool(allow_overwrite=True)
async def gravitational_potential(mass_kg: float, height_m: float) -> str:
    """Compute gravitational potential energy PE = m g h (g = 9.80665 m/s²).

    Args:
        mass_kg: Mass in kilograms
        height_m: Height above reference in meters
    """
    g = 9.80665
    pe = mass_kg * g * height_m
    return f'PE = {mass_kg:g}·{g:g}·{height_m:g} = {pe:.4g} J'


@tool(allow_overwrite=True)
async def equatorial_speed(radius_m: float, period_s: float) -> str:
    """Compute equatorial linear speed v = 2πR / T.

    Args:
        radius_m: Equatorial radius in meters
        period_s: Rotation period in seconds
    """
    v = 2.0 * math.pi * radius_m / period_s
    return f'v = 2πR/T = {v:.2f} m/s ≈ {v * 3.6:.1f} km/h'


def main_llm_cfg() -> dict:
    # repo_id/filename: uses HF hub cache (or ~/models/<filename>) before downloading.
    return {
        'model_type': 'llama_cpp',
        'repo_id': 'Salesforce/xLAM-2-3b-fc-r-gguf',
        'filename': 'xLAM-2-3B-fc-r-F16.gguf',
        'n_ctx': 4096,
        'n_gpu_layers': -1,
        'n_threads': 6,
        'temperature': 0.6,
        'max_tokens': 1024,
        'verbose': False,
        'generate_cfg': {
            'parallel_function_calls': True,
        },
    }


async def main() -> None:
    llm_cfg = main_llm_cfg()
    print(f'LLM: llama_cpp {llm_cfg["repo_id"]} / {llm_cfg["filename"]}')

    bot = Assistant(
        llm=llm_cfg,
        name='PhysicsGuy',
        description='Computes kinetic energy, potential energy, and rotation speeds.',
        system_message=(
            'You are PhysicsGuy. Use the provided tools for every numeric answer. '
            'When the user asks for several independent quantities, call multiple '
            'tools in one turn (parallel function calls). Then summarize briefly.'
        ),
        function_list=['kinetic_energy', 'gravitational_potential', 'equatorial_speed'],
    )

    messages = [
        {
            'role': 'user',
            'content': (
                'For a 2 kg object moving at 3 m/s at a height of 10 m: '
                'what are its kinetic energy and gravitational potential energy? '
                'Also, what is Earth\'s equatorial speed if R = 6378137 m and '
                'T = 86164.0905 s?'
            ),
        }
    ]

    print('\nRunning PhysicsGuy via arun_nonstream '
          '(async path collects full turns; does not stream tokens)...\n')
    async with bot:
        result = await bot.arun_nonstream(messages)

    for msg in result:
        role = msg.get('role') if isinstance(msg, dict) else msg.role
        name = msg.get('name') if isinstance(msg, dict) else msg.name
        content = msg.get('content') if isinstance(msg, dict) else msg.content
        fc = msg.get('function_call') if isinstance(msg, dict) else (
            msg.function_call.model_dump() if getattr(msg, 'function_call', None) else None
        )
        if fc:
            print(f'[{role}] tool_call → {fc}')
        elif role == 'function':
            print(f'[tool:{name}] {content}')
        elif content:
            print(f'[{role}] {content}')


if __name__ == '__main__':
    asyncio.run(main())
