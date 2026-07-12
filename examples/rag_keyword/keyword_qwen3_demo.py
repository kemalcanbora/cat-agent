from pathlib import Path

import torch

from cat_agent.llm.schema import Message, USER
from cat_agent.memory import Memory


def main():
    examples_dir = Path(__file__).parent
    doc_path = examples_dir / "keyword_demo_doc.txt"
    if not doc_path.exists():
        doc_path.write_text(
            "Cat-Agent uses a mandatory Rust BM25 index for keyword retrieval.\n"
            "The index is persisted under workspace/storage/keyword_indexes/ and reused across runs.\n"
            "Python remains the public API for agents, tools, and document records.\n",
            encoding="utf-8",
        )

    llm_cfg = {
        'model': 'Qwen/Qwen3-0.6B',
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

    rag_cfg = {
        'rag_searchers': ['keyword_search'],
        'rebuild_rag': False,
    }
    mem = Memory(llm=llm_cfg, files=[str(doc_path)], rag_cfg=rag_cfg)

    question = "Where is the Rust BM25 index stored and is it reused across runs?"
    messages = [Message(role=USER, content=question)]

    print(f"Question: {question}\n")
    print(f"Using document: {doc_path}\n")

    responses = mem.run_nonstream(messages, force_search=True)

    print("Retrieved knowledge (from Rust keyword-search RAG):")
    print(responses[-1].content)


if __name__ == "__main__":
    main()
