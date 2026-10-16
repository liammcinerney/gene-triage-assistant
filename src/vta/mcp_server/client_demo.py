"""
Minimal MCP client: launches the server as a subprocess over stdio, discovers
its tools, calls them, and probes the input validation.

Run from the repo root: python -m src.vta.mcp_server.client_demo
"""

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "src.vta.mcp_server.server"],
)


def show(label: str, result) -> None:
    status = "ERROR" if result.isError else "ok"
    text = result.content[0].text if result.content else ""
    print(f"\n--- {label} [{status}] ---")
    print(text[:400] + (" ..." if len(text) > 400 else ""))


async def main() -> None:
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print("server:", init.serverInfo.name, "| protocol:", init.protocolVersion)

            # Discovery: the client learns the tools at runtime. Nothing about
            # them is hardcoded on this side.
            tools = await session.list_tools()
            for t in tools.tools:
                print(f"\ntool: {t.name}")
                print("  inputSchema:", json.dumps(t.inputSchema))

            show("gene_lookup TP53", await session.call_tool("gene_lookup", {"symbol": "TP53"}))
            show(
                "literature_search",
                await session.call_tool(
                    "literature_search",
                    {"query": "CFTR modulators rare variants efficacy", "k": 2},
                ),
            )

            # Validation probes: each should be rejected before any API call.
            show("symbol with query operators", await session.call_tool("gene_lookup", {"symbol": "BRCA1 OR TP53"}))
            show("k=500", await session.call_tool("literature_search", {"query": "BRCA1 risk", "k": 500}))
            show("fetch_n smuggled in", await session.call_tool("literature_search", {"query": "BRCA1 risk", "fetch_n": 5000}))


if __name__ == "__main__":
    asyncio.run(main())
