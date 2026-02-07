from pathlib import Path

import torch

from cat_agent.agents import Assistant
from cat_agent.llm.schema import Message
from cat_agent.utils.output_beautify import typewriter_print

LLM_CFG = {
    "model": "Qwen/Qwen3-1.7B",
    "model_type": "transformers",
    "device": "cuda:0" if torch.cuda.is_available() else "mps",
    "generate_cfg": {
        "max_input_tokens": 2048,
        "max_new_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.8,
        "repetition_penalty": 1.2,
    },
}


def main():
    examples_dir = Path(__file__).resolve().parent
    csv_path = examples_dir / "example.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Example CSV not found: {csv_path}")

    agent = Assistant(
        llm=LLM_CFG,
        name="DocParserAgent",
        description="Assistant that can extract and answer questions about document contents (CSV, PDF, DOCX, TXT, etc.).",
        system_message=(
            "You are a helpful assistant. When the user asks about a document, use the simple_doc_parser tool "
            "with the file path they provide (or the path given in the conversation) to extract its content, "
            "then answer based on the extracted text or tables."
        ),
        function_list=["simple_doc_parser"],
        files=[],
    )

    user_content = (
        f"Please use the document parser to read the CSV file at this path: {csv_path}\n"
        "Then tell me: (1) how many products are listed, and (2) which product has the highest total revenue (Price * Units Sold)."
    )
    messages = [Message(role="user", content=user_content)]

    print("User:", user_content.strip(), "\n")
    print("---")
    response_text = ""
    for response in agent.run(messages=messages):
        response_text = typewriter_print(response, response_text)
    print(response_text)


if __name__ == "__main__":
    main()
