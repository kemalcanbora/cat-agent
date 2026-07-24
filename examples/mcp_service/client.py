"""
team-a-agent MCP sunucusu icin basit test istemcisi (FastMCP, SSE).

Once sunucuyu calistir:
    python your_server.py        # http://0.0.0.0:8000 dinler

Sonra:
    pip install mcp
    python test_mcp.py
"""

import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client

# Baglanirken 127.0.0.1 kullan — 0.0.0.0 sadece bind adresidir.
SERVER_URL = "http://127.0.0.1:8000/sse"

# Dumduz topla:
GIRDI = "42 ile 58 i topla"


async def main():
    async with sse_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Mevcut araclar:")
            for t in tools.tools:
                print(f"  - {t.name}")
            print()

            print(f"sum_agent cagriliyor: {GIRDI!r}")
            result = await session.call_tool("sum_agent", {"girdi": GIRDI})

            for block in result.content:
                text = getattr(block, "text", None)
                print("Sonuc:", text if text is not None else block)


if __name__ == "__main__":
    asyncio.run(main())