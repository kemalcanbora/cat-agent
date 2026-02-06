from pathlib import Path
from cat_agent.llm.schema import Message, USER
from cat_agent.memory import Memory
import torch

def main():
    examples_dir = Path(__file__).parent
    doc_path = examples_dir / "leann_demo_doc.txt"
    if not doc_path.exists():
        doc_path.write_text(
            "LEANN saves 97% storage compared to traditional vector databases.\n"
            "This project integrates LEANN as a semantic retriever inside the Qwen agent framework.\n",
            encoding="utf-8",
        )

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

    # RAG configuration: explicitly enable LEANN and ensure its searcher is used.
    rag_cfg = {
        "enable_leann": True,
        "rag_searchers": ["leann_search"],
        "rebuild_rag": False,
    }
    mem = Memory(llm=llm_cfg, files=[str(doc_path)], rag_cfg=rag_cfg)

    question = "How much storage does LEANN save compared to traditional vector databases?"
    messages = [Message(role=USER, content=question)]

    print(f"Question: {question}\n")
    print(f"Using document: {doc_path}\n")

    responses = mem.run_nonstream(messages, force_search=True)

    print("Retrieved knowledge (from LEANN-backed RAG):")
    print(responses[-1].content)


if __name__ == "__main__":
    main()

