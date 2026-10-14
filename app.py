"""AI Patent Research Assistant — Streamlit interface.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html
import logging
import re

import streamlit as st

from patent_rag.config import get_settings
from patent_rag.errors import PatentRAGError
from patent_rag.factory import build_pipeline
from patent_rag.pipeline import ANSWERED, INSUFFICIENT, RECORDED, RETRIEVAL_ONLY, RAGResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="AI Patent Research Assistant", page_icon="🔎", layout="wide")

EXAMPLE_QUESTIONS = [
    "How can a large language model be used to train or improve a smaller language model?",
    "How do the patents use large language models to generate training data for dialogue systems?",
    "How does the attention-free transformer reduce computational cost compared with standard attention?",
    "What does this collection say about CRISPR gene editing in crops?",
]

st.markdown(
    """
    <style>
      .passage {border-left: 3px solid rgba(49,130,206,.6); padding: .6rem .9rem; margin: .4rem 0 .6rem;
                background: rgba(127,127,127,.07); border-radius: 4px; font-size: .92rem; line-height: 1.5;}
      .meta {font-size: .85rem; opacity: .8;}
      .pill {display:inline-block; padding: 1px 8px; border-radius: 10px; font-size: .78rem;
             background: rgba(127,127,127,.15); margin-right: 4px;}
      .pill.cited {background: rgba(56,161,105,.25);}
      .pill.dropped {background: rgba(229,62,62,.2);}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading patent index and embedding model…")
def load_pipeline():
    return build_pipeline(get_settings())


# ---------------------------------------------------------------- startup
settings = get_settings()
try:
    pipeline = load_pipeline()
except PatentRAGError as exc:
    st.title("AI Patent Research Assistant")
    st.error(exc.user_message)
    st.info("Run `python scripts/check_setup.py` for a step-by-step diagnosis.")
    st.stop()

index_size = pipeline.retrievers["semantic"].index.size
patent_ids = pipeline.retrievers["semantic"].index.patent_ids

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Search settings")
    method_label = st.radio(
        "Retrieval method",
        ["Semantic (embeddings)", "Keyword (TF-IDF baseline)"],
        help="Semantic search matches meaning; the keyword baseline matches exact terms. "
        "Switch between them to compare results for the same question.",
    )
    method = "semantic" if method_label.startswith("Semantic") else "keyword"
    top_k = st.slider("Passages to retrieve (top-k)", 1, settings.max_top_k, settings.top_k)
    generate = st.toggle(
        "Generate an AI answer",
        value=True,
        help="Turn off to show retrieval results only (no LLM call).",
    )

    def _label(pid: str) -> str:
        info = pipeline.catalog.get(pid)
        return f"{info.number} — {info.title}" if info.title else info.number

    patent_filter = st.multiselect(
        "Limit to specific patents (optional)", patent_ids, format_func=_label
    )

    st.divider()
    st.subheader("System status")
    st.markdown(f"✅ Index: **{len(patent_ids)} patents**, **{index_size:,} passages**")
    st.markdown(f"✅ Embedding model: `{settings.embedding_model}`")
    if pipeline.llm is not None:
        st.markdown(f"✅ Language model: `{pipeline.llm.model}`")
    else:
        st.markdown("⚠️ Language model: **not configured** (set `OPENAI_API_KEY`). Retrieval still works.")
    st.caption(f"Relevance threshold (semantic): {settings.min_similarity:.2f} cosine similarity")

    st.divider()
    with st.expander("How it works"):
        st.markdown(
            """
1. **Embed** your question into a 384-number vector with `all-MiniLM-L6-v2`.
2. **Retrieve** the top-k most similar patent passages (cosine similarity) from 2,905 pre-embedded chunks.
3. **Gate**: if no passage clears the relevance threshold, the assistant declines instead of guessing.
4. **Generate**: the passages are labeled S1…Sn and sent to the LLM with instructions to answer *only*
   from them and cite labels inline.
5. **Verify**: citations are checked against the passages that were actually sent.
"""
        )
    st.caption(
        "Research aid only — not legal advice, not a prior-art or patentability opinion. "
        "AI-generated answers can be wrong; always check the cited passages."
    )

# ---------------------------------------------------------------- header
st.title("🔎 AI Patent Research Assistant")
st.markdown(
    "Ask natural-language questions about a collection of **33 AI-related US patents and applications**. "
    "The assistant retrieves the most relevant patent passages and writes an answer grounded in them, "
    "with citations you can check."
)

if "question" not in st.session_state:
    st.session_state.question = ""

st.caption("Try an example:")
cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, example in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(example, help=example, use_container_width=True):
        st.session_state.question = example
        st.session_state.run = True

with st.form("ask"):
    st.text_area(
        "Your research question",
        key="question",
        height=80,
        max_chars=settings.max_query_chars,
        placeholder="e.g. How do these patents reduce the cost of attention in transformers?",
    )
    submitted = st.form_submit_button("Search & answer", type="primary")

run = submitted or st.session_state.pop("run", False)


# ---------------------------------------------------------------- rendering helpers
def render_answer_markdown(text: str) -> str:
    # Make inline citations stand out; content stays Markdown (raw HTML is not rendered).
    # Bold whole runs like [S1][S2] at once so adjacent markers don't collide ("****").
    def bold_run(match: re.Match) -> str:
        labels = re.findall(r"S\d+", match.group(0), flags=re.IGNORECASE)
        return "**" + "".join(f"[{label.upper()}]" for label in labels) + "**"

    return re.sub(r"(?:\[\s*S\d+\s*\])+", bold_run, text, flags=re.IGNORECASE)


def render_sources(resp: RAGResponse) -> None:
    evidence_ids = {c.chunk_id for c in resp.evidence}
    cited = set(resp.citations.cited)
    for c in resp.retrieval.chunks:
        info = pipeline.catalog.get(c.patent_id)
        sent = c.chunk_id in evidence_ids
        label = f"S{c.rank}"
        flags = []
        if sent and c.rank in cited:
            flags.append("cited in answer")
        if not sent:
            flags.append("below threshold — not sent to model")
        header = f"{label} · {info.number} · {info.title or 'Title unavailable'} · score {c.score:.3f}"
        if flags:
            header += " · " + ", ".join(flags)
        with st.expander(header, expanded=(c.rank == 1)):
            pills = [f'<span class="pill">rank {c.rank}</span>',
                     f'<span class="pill">{resp.retrieval.method} score {c.score:.3f}</span>']
            if sent and c.rank in cited:
                pills.append('<span class="pill cited">cited</span>')
            if not sent:
                pills.append('<span class="pill dropped">not sent to model</span>')
            meta_bits = [html.escape(info.doc_type or "Patent document")]
            if info.applicant:
                meta_bits.append(f"Applicant: {html.escape(info.applicant)}")
            meta_bits.append(f"Chunk {c.chunk_index} (id <code>{html.escape(c.chunk_id)}</code>)")
            st.markdown(
                " ".join(pills) + f'<div class="meta">{" · ".join(meta_bits)}</div>'
                f'<div class="passage">{html.escape(c.text)}</div>',
                unsafe_allow_html=True,  # all dynamic values above are html-escaped
            )
            st.markdown(
                f"[View on Google Patents ↗]({info.url}) "
                "<span class='meta'>(link constructed from the patent number)</span>",
                unsafe_allow_html=True,
            )


def render_response(resp: RAGResponse) -> None:
    status_text = {
        ANSWERED: ("success", "Answer generated from retrieved patent passages"),
        RECORDED: ("warning", "Recorded answer (not live)"),
        INSUFFICIENT: ("warning", "Insufficient evidence in the patent collection"),
        RETRIEVAL_ONLY: ("info", "Retrieval results only"),
    }
    kind, title = status_text[resp.status]
    getattr(st, kind)(f"**{title}.** {resp.notice or ''}".strip())

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Answer")
        if resp.answer:
            st.markdown(render_answer_markdown(resp.answer))
            caption = f"Generated by `{resp.model}`" if resp.model else ""
            if resp.status == RECORDED:
                caption = f"RECORDED answer from {resp.recorded_at} by `{resp.model}` — not generated live."
            if caption:
                st.caption(caption + " · AI-generated: verify against the cited sources.")
        elif resp.status == RETRIEVAL_ONLY:
            st.markdown("_No generated answer. The retrieved passages are listed on the right and below._")
        else:
            st.markdown("_No answer was generated because the retrieved passages were not relevant enough._")

    with right:
        st.subheader("Retrieval summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Passages retrieved", len(resp.retrieval.chunks))
        m2.metric("Sent to model", len(resp.evidence))
        m3.metric("Top score", f"{resp.retrieval.top_score:.3f}")
        t1, t2, t3 = st.columns(3)
        t1.metric("Retrieval", f"{resp.timings_ms.get('retrieval', 0):.0f} ms")
        gen = resp.timings_ms.get("generation")
        t2.metric("Generation", f"{gen / 1000:.1f} s" if gen else "—")
        t3.metric("Total", f"{resp.timings_ms.get('total', 0) / 1000:.2f} s")
        patents = sorted({c.patent_id for c in resp.retrieval.chunks})
        st.caption(f"Distinct patents in results: {len(patents)} · method: {resp.retrieval.method}")

    tab_sources, tab_how = st.tabs([f"Sources ({len(resp.retrieval.chunks)})", "How this answer was produced"])
    with tab_sources:
        render_sources(resp)
    with tab_how:
        st.markdown(
            f"""
- **Question → vector:** embedded with `{settings.embedding_model}` (semantic) or TF-IDF (keyword).
- **Retrieved:** top {len(resp.retrieval.chunks)} of {index_size:,} passages by {resp.retrieval.method} similarity
  in {resp.timings_ms.get('retrieval', 0):.0f} ms.
- **Relevance gate:** {len(resp.evidence)} passage(s) cleared the threshold
  ({pipeline.threshold(resp.retrieval.method):.2f}) and were sent to the model as sources S1…S{len(resp.evidence)}.
- **Prompt:** system rules require answering only from the sources, citing `[S#]`, flagging insufficient
  evidence, and ignoring instructions inside source text.
- **Citation check:** valid citations {', '.join(f'S{n}' for n in resp.citations.cited) or 'none'}
  {('; invalid: ' + ', '.join(f'S{n}' for n in resp.citations.invalid)) if resp.citations.invalid else ''}.
"""
        )
        if resp.messages:
            with st.expander("Show the exact prompt sent to the language model"):
                for m in resp.messages:
                    st.markdown(f"**{m['role']}**")
                    st.code(m["content"], language="text", wrap_lines=True)


# ---------------------------------------------------------------- run
if run:
    try:
        with st.spinner("Retrieving patent passages and generating an answer…"):
            response = pipeline.answer(
                st.session_state.question,
                top_k=top_k,
                method=method,
                patent_filter=patent_filter or None,
                generate=generate,
            )
        st.session_state.last_response = response
    except PatentRAGError as exc:
        st.session_state.pop("last_response", None)
        st.error(exc.user_message)
    except Exception:  # never show a stack trace in the UI
        logging.getLogger(__name__).exception("unexpected error")
        st.session_state.pop("last_response", None)
        st.error("An unexpected error occurred. Check the terminal logs for details.")

if "last_response" in st.session_state:
    st.divider()
    render_response(st.session_state.last_response)
