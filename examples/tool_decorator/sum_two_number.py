"""Minimal example: define a tool with @tool instead of a BaseTool subclass.

Also ships a zero-arg ``registry()`` + ``agent.yaml`` for API-backed serve/deploy.
"""

from __future__ import annotations

from cat_agent.agents import Assistant
from cat_agent.llm.env_config import llm_config_from_env
from cat_agent.serve import AgentRegistry
from cat_agent.tools import tool


@tool
def sum_two_number(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
        a: First number
        b: Second number
    """
    return a + b


def registry() -> AgentRegistry:
    """Zero-arg factory for serve/deploy. Raises on bad config — never sys.exit."""
    llm = llm_config_from_env(model_type='oai', model='default')
    if 'base_url' in llm and 'model_server' not in llm:
        llm['model_server'] = llm['base_url']
    bot = Assistant(
        llm=llm,
        name='Sum Bot',
        description='Adds two numbers via sum_two_number.',
        function_list=['sum_two_number'],
    )
    reg = AgentRegistry()
    reg.register(bot, name='sum')
    return reg


def main() -> None:
    print('Schema:', sum_two_number.function)
    print('Direct call:', sum_two_number(2, 3))
    print('Via .call():', sum_two_number.call('{"a": 42, "b": 58}'))


if __name__ == '__main__':
    main()
