"""
Example: Agent with a WASM-sandboxed code interpreter.

The agent can write and execute Python code inside a WebAssembly sandbox.
No Docker, no Node.js — just ``pip install wasmtime``.

Setup before running:
    1. pip install wasmtime
    2. Update ``llm_cfg`` below to match your LLM backend.

The WASI CPython binary is bundled with the package — no manual download needed.
"""

from cat_agent.agents import Assistant
import time

def main():
    # -----------------------------------------------------------------
    # LLM config — adjust to your setup (llama_cpp, openai, etc.)
    # -----------------------------------------------------------------
    llm_cfg = {
        'model_type': 'llama_cpp',
        'repo_id': 'Salesforce/xLAM-2-3b-fc-r-gguf',
        'filename': 'xLAM-2-3B-fc-r-F16.gguf',
        'n_ctx': 4096,
        'n_gpu_layers': -1,
        'n_threads': 6,
        'temperature': 0.6,
        'max_tokens': 1024,
        'verbose': False,
    }

    # -----------------------------------------------------------------
    # Create the agent with the WASM code interpreter tool
    # -----------------------------------------------------------------
    bot = Assistant(
        llm=llm_cfg,
        name='Code Runner',
        description='An agent that can write and execute Python code in a secure WASM sandbox.',
        system_message=(
            'You are a helpful coding assistant.  When the user asks you to '
            'compute something, write Python code and execute it using the '
            'wasm_code_interpreter tool.  Only the Python standard library is '
            'available (json, math, re, sqlite3, itertools, collections, '
            'datetime, etc.).'
        ),
        function_list=["wasm_code_interpreter"],
    )

    # -----------------------------------------------------------------
    # Run a few example prompts
    # -----------------------------------------------------------------
    prompts = [
        'What are the first 20 prime numbers? Write Python code to compute them and execute it.',
    ]

    for prompt in prompts:
        print(f'\n{"=" * 60}')
        print(f'USER: {prompt}')
        print('=' * 60)

        messages = [{'role': 'user', 'content': prompt}]

        response = []
        st = time.time()
        for response in bot.run(messages=messages):
            # Print each step so we can see tool calls and results
            last = response[-1]
            role = last.get('role', '')
            if role == 'function':
                print(f'\n[TOOL RESULT] {last.get("name", "")}:')
                print(last.get('content', '')[:500])
            elif role == 'assistant' and last.get('function_call'):
                fc = last['function_call']
                print(f'\n[TOOL CALL] {fc.get("name", "")}')

        # Print final response
        if response:
            final = response[-1]
            if final.get('role') == 'assistant' and not final.get('function_call'):
                print(f'\nASSISTANT:\n{final.get("content", "")}')

        print(time.time() - st, 'seconds elapsed')
        
if __name__ == '__main__':
    main()
