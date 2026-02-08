from cat_agent.agents import Assistant
from cat_agent.llm.schema import ContentItem, Message

# ── LLM configuration ──────────────────────────────────────────────
# Active: Qwen2-VL-2B  (well-tested with llama-cpp-python)
llm_cfg = {
    "model_type": "llama_cpp_vision",
    "repo_id": "ggml-org/Qwen2-VL-2B-Instruct-GGUF",
    "filename": "Qwen2-VL-2B-Instruct-Q8_0.gguf",
    "mmproj_filename": "mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
    "n_ctx": 8192,
    "n_gpu_layers": -1,
    "n_threads": 6,
    "temperature": 0.7,
    "max_tokens": 1024,
    "verbose": False,
}


# ── Image to analyse ───────────────────────────────────────────────
IMAGE_URL = (
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQs4YQpoz5_RRIu8hgeOrbjDZzagpqidQo9Sw&s"
)


def main():
    bot = Assistant(
        llm=llm_cfg,
        name="Vision Bot",
        description="A local vision agent that can describe and analyse images",
    )

    # Construct a multimodal message (text + image)
    messages = [
        Message(
            role="user",
            content=[
                ContentItem(image=IMAGE_URL),
                ContentItem(text="Describe what you see in this image in detail."),
            ],
        )
    ]

    print(f"Image : {IMAGE_URL}")
    print("Prompt: Describe what you see in this image in detail.\n")
    print("Running vision agent …")

    # Stream the response
    response = []
    for response in bot.run(messages=messages):
        print(".", end="", flush=True)

    # Print the final answer
    print("\n\n── Response ──")
    if response:
        last = response[-1]
        content = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
        print(content)


if __name__ == "__main__":
    main()
