import re

import requests

MYGENE_URL = "https://mygene.info/v3/query"

FIELDS = "symbol,name,summary,alias,entrezgene,ensembl.gene,genomic_pos,type_of_gene"

# The symbol is interpolated into a MyGene.info query string, where "*" is a
# wildcard and "OR"/":" are operators. Restricting it to symbol characters keeps
# the query an exact-symbol lookup. The MCP server imports this same pattern for
# its schema, so the agent and MCP paths cannot drift apart.
SYMBOL_PATTERN = r"^[A-Za-z0-9._@-]{1,30}$"


def gene_lookup(symbol: str) -> dict:
    """Look up a human gene by official symbol and return metadata from MyGene.info.

    Args:
        symbol: HGNC gene symbol, e.g. "BRCA1", "TP53". No spaces or wildcards.

    Returns:
        A JSON-serializable dict. On success, "found" is True and the gene
        metadata is included. On failure (invalid symbol, no such gene, network
        error), "found" is False and "error" explains why, so the caller can
        decide how to proceed.
    """
    if not isinstance(symbol, str) or not re.fullmatch(SYMBOL_PATTERN, symbol.strip()):
        return {
            "found": False,
            "error": (
                f"{symbol!r} is not a valid gene symbol. Use one official HGNC symbol "
                "(letters, digits, '-', '.', '_', '@'; max 30 characters), with no "
                "spaces or wildcards."
            ),
        }
    symbol = symbol.strip()

    params = {
        "q": f"symbol:{symbol}",
        "species": "human",
        "fields": FIELDS,
        "size": 1,
    }

    try:
        resp = requests.get(MYGENE_URL, params=params, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"found": False, "error": f"MyGene.info request failed: {e}"}

    hits = resp.json().get("hits", [])
    if not hits:
        return {"found": False, "error": f"No human gene found for symbol '{symbol}'"}

    hit = hits[0]
    # MyGene.info ranks loosely. Accept the hit only if it IS the requested
    # symbol; otherwise we would answer about a different gene than was asked.
    if (hit.get("symbol") or "").upper() != symbol.upper():
        return {
            "found": False,
            "error": (
                f"No human gene has the official symbol '{symbol}' "
                f"(closest match was '{hit.get('symbol')}'). Check the spelling or species."
            ),
        }
    return {
        "found": True,
        "symbol": hit.get("symbol"),
        "name": hit.get("name"),
        "summary": hit.get("summary"),
        "aliases": hit.get("alias"),
        "entrez_id": hit.get("entrezgene"),
        "ensembl_id": (hit.get("ensembl") or {}).get("gene"),
        "genomic_location": hit.get("genomic_pos"),
        "gene_type": hit.get("type_of_gene"),
    }


if __name__ == "__main__":
    import json

    for sym in ["BRCA1", "not_a_real_gene_xyz"]:
        print(f"--- {sym} ---")
        print(json.dumps(gene_lookup(sym), indent=2))
