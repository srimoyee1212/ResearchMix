# researchmix/arxiv_client.py
import re
import uuid
import requests
import streamlit as st
import xml.etree.ElementTree as ET
from typing import Dict, Any, List

from researchmix.text_utils import _strip, _clean_ws

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@st.cache_data(ttl=3600, show_spinner=False)
def arxiv_query(
    search_query: str,
    start: int = 0,
    max_results: int = 10,
    sortBy: str = "submittedDate",
    sortOrder: str = "descending",
) -> List[Dict[str, Any]]:
    """
    Cached arXiv query.
    search_query examples:
      - all:transformer
      - ti:"vision language" AND cat:cs.CV
      - cat:cs.AI OR cat:cs.LG
    """
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }
    r = requests.get(ARXIV_API, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"arXiv API error {r.status_code}: {r.text[:400]}")

    root = ET.fromstring(r.text)
    entries: List[Dict[str, Any]] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = _strip(entry.findtext("atom:id", default="", namespaces=ATOM_NS))
        title = _clean_ws(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        summary = _clean_ws(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
        published = _strip(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
        updated = _strip(entry.findtext("atom:updated", default="", namespaces=ATOM_NS))

        authors: List[str] = []
        for a in entry.findall("atom:author", ATOM_NS):
            name = _strip(a.findtext("atom:name", default="", namespaces=ATOM_NS))
            if name:
                authors.append(name)

        cats: List[str] = []
        for c in entry.findall("atom:category", ATOM_NS):
            term = c.attrib.get("term", "")
            if term:
                cats.append(term)

        paper_id = ""
        if entry_id:
            m = re.search(r"arxiv\.org/abs/([^/]+)$", entry_id)
            paper_id = f"arxiv:{m.group(1)}" if m else f"arxiv:{uuid.uuid4().hex[:10]}"

        year = None
        if published and len(published) >= 4 and published[:4].isdigit():
            year = int(published[:4])

        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break

        entries.append(
            {
                "paper_id": paper_id,
                "title": title or "Untitled (arXiv)",
                "authors": authors,
                "year": year or 2025,
                "abstract": summary,
                "topics": cats,
                "source": "arxiv",
                "arxiv_id": paper_id.replace("arxiv:", ""),
                "url": entry_id,
                "pdf_url": pdf_url,
                "published": published,
                "updated": updated,
            }
        )

    return entries


def arxiv_search_free_text(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    q = _clean_ws(query)
    return arxiv_query(search_query=f"all:{q}", start=0, max_results=max_results)


def arxiv_trending_default(max_results: int = 12) -> List[Dict[str, Any]]:
    cat_query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV"
    return arxiv_query(
        search_query=cat_query,
        start=0,
        max_results=max_results,
        sortBy="submittedDate",
        sortOrder="descending",
    )