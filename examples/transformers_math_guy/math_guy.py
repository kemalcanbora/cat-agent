from cat_agent.utils.output_beautify import typewriter_print
from cat_agent.agents import Assistant
from cat_agent.tools import tool

import torch
from typing import List


@tool
def sum_numbers(numbers: List[float]) -> str:
    """Sum a list of numbers.

    Args:
        numbers: The list of numbers to sum.
    """
    result = sum(numbers)
    return f'The sum of {numbers} is {result}.'


def main():
    llm_cfg = {
        'model': 'Qwen/Qwen3.5-0.8B',
        'model_type': 'transformers',
        'device': 'cuda:0' if torch.cuda.is_available() else 'mps',
        'generate_cfg': {
            'max_input_tokens': 512,
            'max_new_tokens': 128,
            'temperature': 0.3,
            'top_p': 0.8,
            'repetition_penalty': 1.2,
        },
    }


    bot = Assistant(
        llm=llm_cfg,
        name='MathGuy',
        description='A helpful assistant that can answer questions about math and do calculations. '
                    'It can also predict the GPU memory usage of a model based on its architecture and parameters.',
        function_list=["sum_numbers"],
    )

    prompt = '''/no_think
            Sum the following numbers: 1, 2, 3, 4, and 5. Just give me the final answer without any explanation or calculation steps.
    '''

    messages = [{'role': 'user', 'content': prompt}]
    response_text = ''

    for response in bot.run(messages=messages):
        response_text = typewriter_print(response, response_text)
    print(response_text)


if __name__ == "__main__":
    main()
