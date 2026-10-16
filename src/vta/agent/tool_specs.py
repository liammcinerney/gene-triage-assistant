"""
Tool schemas and the name -> function dispatch table, in one place so the
agent loop, MCP server (Stage 4), and eval harness (Stage 5) can all share
a single source of truth instead of redefining tools per-consumer.
"""

from src.vta.tools.gene_lookup import gene_lookup
from src.vta.tools.literature_search import literature_search

GENE_LOOKUP_TOOL = {
    "name": "gene_lookup",
    "description": (
        "Look up metadata for a human gene by its official HGNC symbol "
        "(e.g. 'BRCA1', 'TP53', 'CFTR'). Returns the gene's full name, a "
        "functional summary, known aliases, genomic location, and gene type. "
        "Only accepts official gene symbols, not full gene names, rsIDs, or "
        "protein names. Only covers human genes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Official HGNC gene symbol, e.g. 'BRCA1'.",
            }
        },
        "required": ["symbol"],
    },
}

LITERATURE_SEARCH_TOOL = {
    "name": "literature_search",
    "description": (
        "Find the most relevant PubMed evidence for a specific question or "
        "claim, e.g. 'does hormone replacement therapy increase breast "
        "cancer risk in BRCA1 carriers'. Fetches a pool of candidate papers "
        "from PubMed, then semantically reranks and filters them down to "
        "the chunks actually relevant to the question -- so results are "
        "precise evidence for the specific question asked, not just papers "
        "matching the keywords. Each result includes a similarity score and "
        "a PMID; cite the PMID when using a result to support a claim. Use "
        "this for questions about evidence, recent findings, or published "
        "research -- not for basic gene function, which gene_lookup already "
        "covers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The specific question or claim to find evidence for, "
                    "e.g. 'BRCA1 hormone replacement therapy breast cancer "
                    "risk'. Precise questions retrieve better than broad "
                    "ones."
                ),
            },
            "k": {
                "type": "integer",
                "description": "Max number of evidence chunks to return (default 5).",
            },
        },
        "required": ["query"],
    },
}

TOOLS = [GENE_LOOKUP_TOOL, LITERATURE_SEARCH_TOOL]

TOOL_FUNCTIONS = {
    "gene_lookup": gene_lookup,
    "literature_search": literature_search,
}
