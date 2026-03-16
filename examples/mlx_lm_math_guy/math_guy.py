import sys
from pathlib import Path

# Ensure we import the local repo checkout (not an older site-packages install).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cat_agent.agents import Assistant
from cat_agent.tools.base import BaseTool, register_tool
from cat_agent.utils.output_beautify import typewriter_print


@register_tool("sum_numbers")
class SumNumbers(BaseTool):
    description = "Sum a list of numbers"
    parameters = {
        "type": "object",
        "properties": {
            "numbers": {
                "description": "The list of numbers to sum.",
                "type": "array",
                "items": {"type": "number"},
            }
        },
        "required": ["numbers"],
    }

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        numbers = params["numbers"]
        result = sum(numbers)
        return f"The sum of {numbers} is {result}."


def main():
    llm_cfg = {
        "model_type": "mlx_lm",
        # Any mlx-lm compatible model id or local path.
        "model": "mlx-community/Qwen3.5-0.8B-MLX-8bit",
        "generate_cfg": {
            "max_input_tokens": 512,
            "max_new_tokens": 128,
            "temperature": 0.3,
            "top_p": 0.8,
        },
    }

    bot = Assistant(
        llm=llm_cfg,
        name="MathGuy (MLX)",
        description="A helpful assistant that can answer questions about math and do calculations.",
        function_list=["sum_numbers"],
    )

    prompt = """/no_think
Sum the following numbers: 1, 2, 3, 4, and 5. Just give me the final answer without any explanation or calculation steps.
"""

    messages = [{"role": "user", "content": prompt}]
    response_text = ""
    for response in bot.run(messages=messages):
        response_text = typewriter_print(response, response_text)


if __name__ == "__main__":
    main()

