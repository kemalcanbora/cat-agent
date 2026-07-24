"""Graph workflow example, MathGuy edition.

Same model and `sum_numbers` tool as examples/transformers_math_guy/math_guy.py,
but wired as a graph instead of a single Assistant:

    start -> classify --(math)--> math_guy --> END
                       \--(else)--> chat -----> END

`classify` is a plain Python node that sets a routing flag; the conditional edge
picks the branch. The compiled graph is itself an Agent, so `bot.run(messages)`
works exactly like the other examples.
"""

import os

from typing import List

from cat_agent.agents import Assistant
from cat_agent.graph import END, AgentNode, FunctionNode, StateGraph
from cat_agent.graph.state import GraphState
from cat_agent.observability import MermaidExporter, PrintHandler
from cat_agent.tools import tool
from cat_agent.utils.output_beautify import typewriter_print

import torch


@tool
def sum_numbers(numbers: List[float]) -> str:
    """Sum a list of numbers.

    Args:
        numbers: The list of numbers to sum.
    """
    result = sum(numbers)
    return f'The sum of {numbers} is {result}.'


LLM_CFG = {
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


def classify(state: GraphState) -> GraphState:
    """Route to the math branch when the question looks numeric."""
    text = state.last_message.content or ''
    text = text if isinstance(text, str) else str(text)
    lowered = text.lower()
    math_words = ('sum', 'add', 'plus', 'total', 'calculate', 'multiply', 'average')
    state.scratch['is_math'] = any(c.isdigit() for c in text) or any(w in lowered for w in math_words)
    return state


def build_graph():
    math_guy = Assistant(
        llm=LLM_CFG,
        name='MathGuy',
        description='A helpful assistant that can answer questions about math and do calculations.',
        function_list=['sum_numbers'],
    )
    chat = Assistant(
        llm=LLM_CFG,
        name='Chat',
        description='A friendly general-purpose assistant for non-math questions.',
        system_message='You are a friendly assistant. Keep replies brief.',
    )

    graph = (
        StateGraph()
        .add_node(FunctionNode('classify', classify))
        .add_node(AgentNode('math_guy', math_guy))
        .add_node(AgentNode('chat', chat))
        .set_entry('classify')
        .add_conditional_edges('classify', lambda s: 'math_guy' if s.scratch['is_math'] else 'chat')
        .add_edge('math_guy', END)
        .add_edge('chat', END)
    )
    # GRAPH_TRACE=1 prints node.start/node.end events AND writes a Mermaid diagram
    # of the DAG path taken to graph_dag.mmd (paste into https://mermaid.live or any
    # Mermaid-aware viewer). Forced on here for the demo.
    os.environ['GRAPH_TRACE'] = '1'

    handlers = None
    if os.getenv('GRAPH_TRACE'):
        handlers = [PrintHandler(), MermaidExporter(path='graph_dag.mmd')]
    return graph.compile(name='MathGuyGraph', handlers=handlers)


def main():
    bot = build_graph()

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
