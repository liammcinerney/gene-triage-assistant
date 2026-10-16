import requests

MYGENE_URL = "https://mygene.info/v3/query"

FIELDS = "symbol,name,summary,alias,entrezgene,ensembl.gene,genomic_pos,type_of_gene"


def gene_lookup(symbol: str) -> dict:
    """Look up a human gene by symbol and return summary metadata from MyGene.info.

    Args:
        symbol: HGNC gene symbol, e.g. "BRCA1", "TP53".

    Returns:
        A JSON-serializable dict. On success, "found" is True and the gene
        metadata is included. On failure, "found" is False and "error"
        explains why, so the caller can decide how to proceed.
    """
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
