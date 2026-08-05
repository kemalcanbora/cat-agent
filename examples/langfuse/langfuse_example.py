"""Send Cat-Agent traces to a local Langfuse instance via OpenTelemetry.

Prerequisites
-------------
1. Start Langfuse::

       cd examples/langfuse && docker compose up -d

2. Copy env and install the OTel extra::

       cp examples/langfuse/.env.example examples/langfuse/.env
       pip install 'cat-agent[otel]'

Run::

    python examples/langfuse/langfuse_example.py

Sign in at http://localhost:3000 with ``demo@example.com`` / ``password1``
(credentials from ``docker-compose.yaml`` ``LANGFUSE_INIT_*``).
"""

from __future__ import annotations
import time
from pathlib import Path

from cat_agent.agents import Assistant
from cat_agent.llm.schema import USER, Message
from cat_agent.observability import OpenTelemetryHandler, PrintHandler, with_langfuse
from cat_agent.tools import tool

_ENV_FILE = Path(__file__).resolve().parent / '.env'
if not _ENV_FILE.is_file():
    _ENV_FILE = Path(__file__).resolve().parent / '.env.example'


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name
    """
    time.sleep(0.2)
    return f'The weather in {city} is 22 C and sunny.'


@with_langfuse(env_file=_ENV_FILE)
def main() -> None:
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
        handlers=[
            OpenTelemetryHandler(),
            PrintHandler(),
        ],
    )

    messages = [Message(role=USER, content='What is the weather in Istanbul?')]
    response = []
    for response in bot.run(messages):
        pass

    if response:
        print('\nFinal reply:', response[-1].get('content', response[-1].content))


if __name__ == '__main__':
    main()
