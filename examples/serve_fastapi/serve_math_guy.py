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

"""API-backed calculator agent for Nomad / ``cat-agent serve``.

Local llama_cpp demo: see ``examples/llama_cpp_math_guy/`` (no agent.yaml).

    pip install 'cat-agent[serve,platform]'
    python examples/serve_fastapi/serve_math_guy.py

    cat-agent deploy --dir examples/serve_fastapi
"""
from __future__ import annotations

from pathlib import Path

from cat_agent.agents import Assistant
from cat_agent.llm.env_config import llm_config_from_env
from cat_agent.serve import AgentRegistry, create_app, run_app
from cat_agent.tools import tool


@tool(allow_overwrite=True)
def sum_two_number(a: float, b: float) -> str:
    """Add two numbers together. Provide 'a' and 'b' as numbers.

    Args:
        a: The first number
        b: The second number
    """
    return f'The sum of {a} and {b} is {a + b}.'


def registry() -> AgentRegistry:
    """Zero-arg factory for ``cat-agent serve`` / Nomad deploy.

    Raises on bad config — never ``sys.exit`` (lifespan reports via /readyz).
    Model comes from env (Nomad injects ``CAT_AGENT_LLM_MODEL`` from agent.yaml)
    or, locally, from sibling ``agent.yaml`` when no model env var is set.
    """
    llm = llm_config_from_env(
        model_type='oai',
        agent_yaml=Path(__file__).resolve().parent / 'agent.yaml',
    )
    bot = Assistant(
        llm=llm,
        name='Calculator Bot',
        description='An agent that can sum two numbers.',
        function_list=['sum_two_number'],
    )
    reg = AgentRegistry()
    reg.register(bot, name='calculator')
    return reg


def main() -> None:
    app = create_app(registry())
    print('Serving calculator (see CAT_AGENT_SERVE_HOST/PORT)')
    print('  POST /agents/calculator/run')
    run_app(app)


if __name__ == '__main__':
    main()
