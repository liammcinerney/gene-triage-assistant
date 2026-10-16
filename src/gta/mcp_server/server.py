"""
MCP server exposing the Gene Triage tools over stdio.

This process makes no LLM calls and needs no API key: it only serves tools.
The *client's* model decides when to call them. Run it directly with
`python -m src.gta.mcp_server.server`, or let an MCP client launch it.

stdio rule: stdout carries the JSON-RPC protocol, so nothing in this process
may print to stdout. Log to stderr only.
"""

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from src.gta.agent.tool_specs import GENE_LOOKUP_TOOL, LITERATURE_SEARCH_TOOL
from src.gta.tools.gene_lookup import SYMBOL_PATTERN
from src.gta.tools.gene_lookup import gene_lookup as _gene_lookup
from src.gta.tools.literature_search import MAX_K, MAX_QUERY_LEN, MIN_K, MIN_QUERY_LEN
from src.gta.tools.literature_search import literature_search as _literature_search

mcp = FastMCP("gene-triage")

# Hints for clients, not security guarantees: both tools only read, and both
# reach out to external services (MyGene.info, NCBI).
READ_ONLY_EXTERNAL = ToolAnnotations(readOnlyHint=True, openWorldHint=True)

# Descriptions come from tool_specs.py so the wording that steers the model
# (prompt engineering: it is the only thing the model sees of a tool) has
# exactly one source of truth.
_symbol_desc = GENE_LOOKUP_TOOL["input_schema"]["properties"]["symbol"]["description"]
_query_desc = LITERATURE_SEARCH_TOOL["input_schema"]["properties"]["query"]["description"]


@mcp.tool(
    name=GENE_LOOKUP_TOOL["name"],
    description=GENE_LOOKUP_TOOL["description"],
    annotations=READ_ONLY_EXTERNAL,
)
def gene_lookup(
    # Constraints are advertised in the schema so a client's model sees them up
    # front; the tool function enforces the same rules itself (single source:
    # SYMBOL_PATTERN), so the agent path is protected too.
    symbol: Annotated[str, Field(description=_symbol_desc, pattern=SYMBOL_PATTERN)],
) -> dict:
    return _gene_lookup(symbol)


@mcp.tool(
    name=LITERATURE_SEARCH_TOOL["name"],
    description=LITERATURE_SEARCH_TOOL["description"],
    annotations=READ_ONLY_EXTERNAL,
)
def literature_search(
    query: Annotated[str, Field(description=_query_desc, min_length=MIN_QUERY_LEN, max_length=MAX_QUERY_LEN)],
    k: Annotated[int, Field(description="Max number of papers to return.", ge=MIN_K, le=MAX_K)] = 5,
) -> dict:
    # fetch_n is deliberately not a parameter here. FastMCP builds the public
    # schema from this signature, so exposing it would let any client request
    # thousands of PubMed candidates. The server owns that cost knob.
    return _literature_search(query, k=k)


if __name__ == "__main__":
    mcp.run(transport="stdio")
