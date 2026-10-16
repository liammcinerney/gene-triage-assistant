"""
MCP server exposing the Variant Triage tools over stdio.

This process makes no LLM calls and needs no API key: it only serves tools.
The *client's* model decides when to call them. Run it directly with
`python -m src.vta.mcp_server.server`, or let an MCP client launch it.

stdio rule: stdout carries the JSON-RPC protocol, so nothing in this process
may print to stdout. Log to stderr only.
"""

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from src.vta.agent.tool_specs import GENE_LOOKUP_TOOL, LITERATURE_SEARCH_TOOL
from src.vta.tools.gene_lookup import gene_lookup as _gene_lookup
from src.vta.tools.literature_search import literature_search as _literature_search

mcp = FastMCP("variant-triage")

# Hints for clients, not security guarantees: both tools only read, and both
# reach out to external services (MyGene.info, NCBI).
READ_ONLY_EXTERNAL = ToolAnnotations(readOnlyHint=True, openWorldHint=True)

# Descriptions come from tool_specs.py so the wording that steers the model
# (prompt engineering, see Stage 1) has exactly one source of truth.
_symbol_desc = GENE_LOOKUP_TOOL["input_schema"]["properties"]["symbol"]["description"]
_query_desc = LITERATURE_SEARCH_TOOL["input_schema"]["properties"]["query"]["description"]


@mcp.tool(
    name=GENE_LOOKUP_TOOL["name"],
    description=GENE_LOOKUP_TOOL["description"],
    annotations=READ_ONLY_EXTERNAL,
)
def gene_lookup(
    # The pattern is a security control, not tidiness: the symbol is
    # interpolated into a MyGene.info query string ("symbol:<value>"), so
    # an unrestricted value like "BRCA1 OR TP53" would rewrite the query.
    symbol: Annotated[
        str,
        Field(description=_symbol_desc, min_length=1, max_length=30, pattern=r"^[A-Za-z0-9._@-]+$"),
    ],
) -> dict:
    return _gene_lookup(symbol)


@mcp.tool(
    name=LITERATURE_SEARCH_TOOL["name"],
    description=LITERATURE_SEARCH_TOOL["description"],
    annotations=READ_ONLY_EXTERNAL,
)
def literature_search(
    query: Annotated[str, Field(description=_query_desc, min_length=3, max_length=300)],
    k: Annotated[int, Field(description="Max number of evidence chunks to return.", ge=1, le=10)] = 5,
) -> dict:
    # fetch_n is deliberately not a parameter here. FastMCP builds the public
    # schema from this signature, so exposing it would let any client request
    # thousands of PubMed candidates. The server owns that cost knob.
    return _literature_search(query, k=k)


if __name__ == "__main__":
    mcp.run(transport="stdio")
