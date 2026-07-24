from mcp.server.fastmcp import FastMCP
from cat_agent.agents import Assistant
from cat_agent.tools import tool
import torch


@tool
def sum_two_number(a: float, b: float) -> str:
    """İki sayıyı toplar. 'a' ve 'b' sayılarını verin.

    Args:
        a: Birinci sayı
        b: İkinci sayı
    """
    return f'{a} ile {b} sayısının toplamı {a + b}.'


# --- A takımının ajanı (X modeli) ---
a_agent = Assistant(
    llm={
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
        "model_server": "dashscope",
    },
    name="Toplama Ajani",
    description="Toplama işlemleri yapan ajan.",
    function_list=["sum_two_number"],
    system_message="Sen bir toplama uzmanısın. sum_two_number aracını kullanarak topla.",
)


# --- Ajanı MCP üzerinden dışarı aç ---
mcp = FastMCP("team-a-agent", host="0.0.0.0", port=8000)


@mcp.tool()
def sum_agent(girdi: str) -> str:
    """A takımının toplama ajanı. Doğal dil ile verilen sayıları toplar.
    Örnek girdi: '42 ile 58 i topla'. Sonucu metin olarak döner."""
    messages = [{"role": "user", "content": girdi}]
    son = ""
    for response in a_agent.run(messages=messages):
        if response:
            son = response[-1].get("content", "")
    return son


if __name__ == "__main__":
    mcp.run(transport="sse")
