from cat_agent.utils.output_beautify import typewriter_print
from cat_agent.agents import Assistant
from cat_agent.tools.base import BaseTool, register_tool

import torch


@register_tool('sum_numbers')
class SumNumbers(BaseTool):
    description = """Sum a list of numbers"""

    parameters = {
        'type': 'object',
        'properties': {
            'numbers': {
                'description': 'The list of numbers to sum.',
                'type': 'array',
                'items': {
                    'type': 'number'
                }
            }
        },
        'required': ['numbers']
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        numbers = params['numbers']
        result = sum(numbers)
        return f"The sum of {numbers} is {result}."


def main():
    llm_cfg = {
        'model': 'Qwen/Qwen3-1.7B',
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

    prompt = f'''/no_think
            Sum the following numbers: 1, 2, 3, 4, and 5. Just give me the final answer without any explanation or calculation steps.
    '''

    messages = [{'role': 'user', 'content': prompt}]
    response_text = ''

    for response in bot.run(messages=messages):
        response_text = typewriter_print(response, response_text)
    print(response_text)


if __name__ == "__main__":
    main()

