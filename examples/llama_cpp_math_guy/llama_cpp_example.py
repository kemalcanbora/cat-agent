from cat_agent.agents import Assistant
from cat_agent.serve import AgentRegistry
from cat_agent.tools import tool


@tool
def sum_two_number(a: float, b: float) -> str:
    """Add two numbers together. Provide 'a' and 'b' as numbers.

    Args:
        a: The first number
        b: The second number
    """
    return f'The sum of {a} and {b} is {a + b}.'


def registry() -> AgentRegistry:
    """Local-only factory for ``cat-agent serve`` (no agent.yaml / Nomad deploy)."""
    llm_cfg = {
        'model_type': 'llama_cpp',
        'repo_id': 'Salesforce/xLAM-2-3b-fc-r-gguf',
        'filename': 'xLAM-2-3B-fc-r-F16.gguf',
        'n_ctx': 4096,
        'n_gpu_layers': -1,
        'n_threads': 6,
        'temperature': 0.6,
        'max_tokens': 1024,
        'verbose': False,
    }
    bot = Assistant(
        llm=llm_cfg,
        name='Calculator Bot',
        description='An agent that can sum two numbers.',
        function_list=['sum_two_number'],
    )
    reg = AgentRegistry()
    reg.register(bot, name='calculator')
    return reg


def main():
    # repo_id/filename: uses HF hub cache (or ~/models/<filename>) before downloading.
    bot = registry().get('calculator')

    messages = [
        {'role': 'user',
         'content': 'Please sum the numbers 42 and 58 for me.'}
    ]

    print('\nRunning agent...')
    response = []
    for response in bot.run(messages=messages):
        print('.', end='', flush=True)

    # Print final response
    print('\n\nFinal response:')
    if response:
        print(response[-1].get('content', ''))


if __name__ == "__main__":
    main()
