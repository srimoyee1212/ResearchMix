# researchmix/ui/graph.py
from typing import Dict, Any, List, Tuple
import streamlit as st

from streamlit_agraph import agraph, Node, Edge, Config

from researchmix.state import u_bucket, get_paper


def _topic_overlap(a: List[str], b: List[str]) -> int:
    sa = set([x.strip() for x in (a or []) if x])
    sb = set([x.strip() for x in (b or []) if x])
    return len(sa.intersection(sb))


def _candidate_pool(limit_per_list: int = 15) -> List[str]:
    """
    Papers to consider as graph neighbors.
    We keep it lightweight + local: recommended + trending + history.
    """
    bucket = u_bucket()
    ids: List[str] = []
    ids.extend((bucket.get("playlists", {}).get("recommended") or [])[:limit_per_list])
    ids.extend((bucket.get("playlists", {}).get("trending") or [])[:limit_per_list])
    ids.extend((bucket.get("history") or [])[:limit_per_list])

    # de-dupe, keep order
    seen = set()
    out = []
    for pid in ids:
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def render_related_paper_graph(
    center_paper_id: str,
    *,
    max_neighbors: int = 10,
    min_overlap: int = 1,
    height: int = 520,
):
    """
    Simple graph:
      - Center node = selected paper
      - Neighbor nodes = papers with overlapping arXiv categories/topics
      - Edge weight = overlap count (displayed as edge label)
      - Clicking a node opens it (via Streamlit agraph return value)
    """
    center = get_paper(center_paper_id)
    if not center:
        st.info("Select a paper to see its related graph.")
        return

    center_topics = center.get("topics") or []
    if not center_topics:
        st.info("This paper has no topics/categories, so the graph can’t be built.")
        return

    candidates = [pid for pid in _candidate_pool() if pid != center_paper_id]
    scored: List[Tuple[str, int]] = []
    for pid in candidates:
        p = get_paper(pid)
        if not p:
            continue
        ov = _topic_overlap(center_topics, p.get("topics") or [])
        if ov >= min_overlap:
            scored.append((pid, ov))

    scored.sort(key=lambda x: x[1], reverse=True)
    neighbors = scored[:max_neighbors]

    if not neighbors:
        st.info("No closely related papers found (based on topic overlap).")
        return

    # ---- Nodes
    nodes: List[Node] = []
    edges: List[Edge] = []

    def short_title(t: str, n: int = 42) -> str:
        t = (t or "").strip()
        if len(t) <= n:
            return t
        return t[: n - 1].rstrip() + "…"

    # Center node
    nodes.append(
        Node(
            id=center_paper_id,
            label=short_title(center.get("title", "Selected paper")),
            size=28,
            title="Selected paper",
        )
    )

    # Neighbor nodes
    for pid, ov in neighbors:
        p = get_paper(pid) or {}
        title = p.get("title", "Untitled")
        year = p.get("year", "")
        source = p.get("source", "")
        topics = ", ".join((p.get("topics") or [])[:6])

        hover = f"{title}\n{year} · {source}\nOverlap: {ov}\nTopics: {topics}"
        nodes.append(
            Node(
                id=pid,
                label=short_title(title),
                size=18 + min(ov, 6),  # slightly larger for higher overlap
                title=hover,
            )
        )
        edges.append(
            Edge(
                source=center_paper_id,
                target=pid,
                label=str(ov),
                value=ov,
            )
        )

    config = Config(
        width="100%",
        height=height,
        directed=False,
        physics=True,
        hierarchical=False,
        # Let users click nodes
        collapsible=False,
    )

    st.caption("Edges show **# shared topics** (arXiv categories). Click a node to open it.")
    clicked = agraph(nodes=nodes, edges=edges, config=config)

    # streamlit-agraph returns the clicked node id (or a dict depending on version)
    selected_id = None
    if isinstance(clicked, str):
        selected_id = clicked
    elif isinstance(clicked, dict):
        # some versions return {"id": "..."} or {"node": "..."}
        selected_id = clicked.get("id") or clicked.get("node")

    if selected_id and selected_id != center_paper_id:
        # Update selection and jump to Paper view
        bucket = u_bucket()
        bucket["selected_paper"] = selected_id
        bucket["last_played"] = selected_id
        from researchmix.state import bump_history, goto

        bump_history(selected_id)
        goto("Paper")