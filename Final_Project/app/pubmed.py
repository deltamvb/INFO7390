from typing import Dict, List
import xml.etree.ElementTree as ET

import requests


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed_ids(query: str, max_results: int = 50) -> List[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance",
    }
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("esearchresult", {}).get("idlist", [])


def fetch_pubmed_abstracts(pmids: List[str]) -> List[Dict]:
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    resp = requests.get(EFETCH_URL, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    docs = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_parts = article.findall(".//Abstract/AbstractText")

        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        abstract = " ".join(
            part.text.strip() for part in abstract_parts if part.text and part.text.strip()
        )

        if abstract:
            docs.append(
                {
                    "id": f"pmid-{pmid}",
                    "pmid": pmid,
                    "title": title,
                    "text": abstract,
                    "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )
    return docs
