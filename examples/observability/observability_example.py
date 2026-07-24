"""Demonstrate Cat-Agent observability hooks with a runnable agent.

    python examples/observability/observability_example.py

Uses a local GGUF model (same as logging_demo). Observability events print
via CallbackHandler — no manual event-type parsing needed.
"""

from __future__ import annotations

import time

from cat_agent.agents import Assistant
from cat_agent.llm.schema import USER, Message
from cat_agent.observability import CallbackHandler
from cat_agent.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name
    """
    time.sleep(0.2)
    return f'The weather in {city} is 22 C and sunny.'


def on_event(event):
    """No parsing — event.summary() formats every event type."""
    print(event.summary())


def main():
    # repo_id/filename: uses HF hub cache (or ~/models/<filename>) before downloading.
    llm_cfg = {
        'model_type': 'llama_cpp',
        'repo_id': 'Salesforce/xLAM-2-3b-fc-r-gguf',
        'filename': 'xLAM-2-3B-fc-r-F16.gguf',
        'n_ctx': 4096,
        'n_gpu_layers': -1,
        'n_threads': 6,
        'temperature': 0.6,
        'max_tokens': 512,
        'verbose': False,
    }

    bot = Assistant(
        llm=llm_cfg,
        name='WeatherBot',
        description='Looks up weather for a city.',
        function_list=['get_weather'],
        handlers=[CallbackHandler(on_event)],
    )

    messages = [Message(role=USER, content='What is the weather in Istanbul?')]
    print('Observability events:\n')
    response = []
    for response in bot.run(messages):
        pass

    if response:
        print('\nFinal reply:', response[-1].get('content', response[-1].content))


if __name__ == '__main__':
    main()
