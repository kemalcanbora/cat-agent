import torch
from cat_agent.agents import Assistant, Router
from cat_agent.llm.schema import Message
from cat_agent.utils.output_beautify import typewriter_print

LLM_CFG = {
    "model": "Qwen/Qwen3-1.7B",
    "model_type": "transformers",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "generate_cfg": {
        "max_input_tokens": 512,
        "max_new_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.8,
        "repetition_penalty": 1.2,
    },
}


def build_router():
    llm_cfg = LLM_CFG

    math_expert = Assistant(
        llm=llm_cfg,
        name="MathExpert",
        description="Expert in mathematics: arithmetic, algebra, equations, and step-by-step calculation.",
        system_message="You are a math expert. Answer concisely with clear steps when helpful.",
    )

    general_assistant = Assistant(
        llm=llm_cfg,
        name="GeneralAssistant",
        description="General-purpose assistant for greetings, casual chat, and non-math questions.",
        system_message="You are a friendly general assistant. Keep replies helpful and brief.",
    )

    router = Router(
        llm=llm_cfg,
        name="Router",
        description="Coordinates user requests and delegates to specialists when needed.",
        agents=[math_expert, general_assistant],
    )
    return router


def main():
    router = build_router()

    # Example 1: Should be delegated to MathExpert
    messages_math = [Message(role="user", content="What is 17 * 23? Show the calculation.")]
    print("User: What is 17 * 23? Show the calculation.")
    print("---")
    response_text = ""
    for response in router.run(messages=messages_math):
        response_text = typewriter_print(response, response_text)
    print(response_text)
    print()

    # Example 2: Router may answer directly or use GeneralAssistant
    messages_hello = [Message(role="user", content="Hi, what can you help me with?")]
    print("User: Hi, what can you help me with?")
    print("---")
    response_text = ""
    for response in router.run(messages=messages_hello):
        response_text = typewriter_print(response, response_text)
    print(response_text)


if __name__ == "__main__":
    main()
