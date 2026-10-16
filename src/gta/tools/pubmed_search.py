import xml.etree.ElementTree as ET

import requests

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

NO_ABSTRACT = "(no abstract available)"


def pubmed_search(query: str, max_results: int = 5) -> dict:
    """Search PubMed and return titles + abstracts for the top matching papers.

    Two calls under the hood: esearch (query -> list of PMIDs), then efetch
    (PMIDs -> full abstract XML). NCBI doesn't offer a single-call endpoint
    that returns both IDs and abstract text.

    Args:
        query: Free-text search query, e.g. "BRCA1 breast cancer risk".
        max_results: Maximum number of papers to return.

    Returns:
        A JSON-serializable dict. On success, "found" is True and "papers"
        is a list of {pmid, title, abstract, journal, year}. On failure or
        zero results, "found" is False and "error" explains why.
    """
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    try:
        search_resp = requests.get(ESEARCH_URL, params=search_params, timeout=10)
        search_resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"found": False, "error": f"PubMed search request failed: {e}"}

    pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return {"found": False, "error": f"No PubMed results for query '{query}'"}

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    try:
        fetch_resp = requests.get(EFETCH_URL, params=fetch_params, timeout=15)
        fetch_resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"found": False, "error": f"PubMed fetch request failed: {e}"}

    papers = _parse_pubmed_xml(fetch_resp.text)
    return {"found": True, "count": len(papers), "papers": papers}


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")

        # Use itertext(), not .text: titles and abstracts often wrap gene
        # symbols in <i>/<sup> tags (e.g. "<ArticleTitle>Role of <i>BRCA1</i>
        # in...</ArticleTitle>"). .text only grabs text before the first
        # nested tag and silently drops everything after it.
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""

        abstract_parts = [
            "".join(el.itertext())
            for el in article.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(part.strip() for part in abstract_parts if part.strip())

        journal = article.findtext(".//Journal/Title", default="")
        year = article.findtext(".//JournalIssue/PubDate/Year", default="")

        papers.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract or NO_ABSTRACT,
                "journal": journal,
                "year": year,
            }
        )
    return papers


if __name__ == "__main__":
    import json

    print(json.dumps(pubmed_search("BRCA1 breast cancer risk", max_results=3), indent=2))
