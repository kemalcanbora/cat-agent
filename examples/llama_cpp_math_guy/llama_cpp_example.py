from cat_agent.agents import Assistant
from cat_agent.tools.base import BaseTool, register_tool


@register_tool('sum_two_number')
class SumTwoNumber(BaseTool):
    description = "Add two numbers together. Provide 'a' and 'b' as numbers."
    parameters = {
        'type': 'object',
        'properties': {
            'a': {'description': 'The first number', 'type': 'number'},
            'b': {'description': 'The second number', 'type': 'number'},
        },
        'required': ['a', 'b'],
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        try:
            a = float(params["a"])
            b = float(params["b"])
        except (KeyError, ValueError, TypeError):
            return "Error: Please provide valid numbers for 'a' and 'b'."
        total = a + b
        return f"The sum of {a} and {b} is {total}."

def main():
    # Use a local GGUF file, or repo_id + filename for HuggingFace
    llm_cfg = {
        "model_type": "llama_cpp",
        "repo_id": "Salesforce/xLAM-2-3b-fc-r-gguf",
        "filename": "xLAM-2-3B-fc-r-F16.gguf",
        "n_ctx": 4096,
        "n_gpu_layers": -1,
        "n_threads": 6,
        "temperature": 0.6,
        "max_tokens": 1024,
        "verbose": False,
    }

    bot = Assistant(
        llm=llm_cfg,
        name="Calculator Bot",
        description='An agent that can sum two numbers.',
        function_list=["sum_two_number"],
    )

    messages = [
        {"role": "user",
         "content": "Please sum the numbers 42 and 58 for me."}
    ]

    print("\nRunning agent...")
    response = []
    for response in bot.run(messages=messages):
        print(".", end="", flush=True)

    # Print final response
    print("\n\nFinal response:")
    if response:
        print(response[-1].get('content', ''))


if __name__ == "__main__":
    main()
