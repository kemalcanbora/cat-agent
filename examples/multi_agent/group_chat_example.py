import torch
from cat_agent.agents import Assistant, GroupChat
from cat_agent.llm.schema import Message
from cat_agent.utils.output_beautify import typewriter_print

LLM_CFG = {
    "model": "Qwen/Qwen3-1.7B",
    "model_type": "transformers",
    "device": "cuda:0" if torch.cuda.is_available() else "mps",
    "generate_cfg": {
        "max_input_tokens": 512,
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.8,
        "repetition_penalty": 1.2,
    },
}


def build_group_chat():
    """Build a GroupChat with two agents speaking in turn."""
    llm_cfg = LLM_CFG

    alice = Assistant(
        llm=llm_cfg,
        name="Alice",
        description="Friendly and curious. Likes to ask follow-up questions.",
        system_message="You are Alice. Be concise and a bit curious.",
    )

    bob = Assistant(
        llm=llm_cfg,
        name="Bob",
        description="Practical and direct. Prefers short answers.",
        system_message="You are Bob. Keep replies short and to the point.",
    )

    group = GroupChat(
        agents=[alice, bob],
        agent_selection_method="round_robin",
        name="GroupChat",
    )
    return group


def main():
    group = build_group_chat()

    messages = [
        Message(role="user",
                content="Let's plan a short weekend trip. Ideas?",
                name="user"),
    ]

    print("User: Let's plan a short weekend trip. Ideas?")
    print("---")

    response_text = ""
    max_round = 3
    for response in group.run(messages=messages, max_round=max_round):
        response_text = typewriter_print(response, response_text)

    print(response_text)


if __name__ == "__main__":
    main()
