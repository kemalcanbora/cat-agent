"""Minimal example: define a tool with @tool instead of a BaseTool subclass."""

from cat_agent.tools import tool


@tool
def sum_two_number(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
        a: First number
        b: Second number
    """
    return a + b


def main() -> None:
    print('Schema:', sum_two_number.function)
    print('Direct call:', sum_two_number(2, 3))
    print('Via .call():', sum_two_number.call('{"a": 42, "b": 58}'))

    # Optional: wire into an Assistant when an LLM is configured.
    # from cat_agent.agents import Assistant
    # bot = Assistant(llm={...}, function_list=['sum_two_number'])
    # # or: function_list=[sum_two_number]


if __name__ == '__main__':
    main()
