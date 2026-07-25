"""
app/main.py 

Streamlit frontend for the PubMed RAG Research Assistant.

Run:
    streamlit run app/main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st
from uvicorn import Config

ProjectRoot = Path(__file__).parent.parent
if ProjectRoot not in sys.path:
    sys.path.append(0, str(ProjectRoot))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname) - 8s | %(name)s | %(message)s",

)
logger = logging.getLogger(__name__)

# Page Config 

st.set_page_config(
    page_title="PubMed RAG Research Assistant",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/pubmed-rag-system",
        "Report a bug": "None",
        "About": (
            "PubMed RAG Research Assistant\n"
            "Real-time literature synthesis powered by "
            "PubMedBERT + FlashRank + Claude."
        ), 
    },
)

# Global CSS:

st.markdown("""
<style>
/* ── Header banner ── */
.rag-header {
    background: linear-gradient(135deg, #0d2b45 0%, #1565c0 100%);
    padding: 1.8rem 2rem;
    border-radius: 12px;
    color: #ffffff;
    margin-bottom: 1.5rem;
}
.rag-header h1 { margin: 0; font-size: 1.9rem; }
.rag-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.9rem; }

/* ── Paper card ── */
.paper-card {
    border: 1px solid #dde3ec;
    border-left: 4px solid #1565c0;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    background: #f8fafc;
    line-height: 1.55;
}
.paper-card a { color: #1565c0; text-decoration: none; font-weight: 600; }
.paper-card a:hover { text-decoration: underline; }

/* ── Metric card ── */
.metric-box {
    background: #f0f4fb;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

/* ── Code block query display ── */
.stCodeBlock { font-size: 0.82rem; }
</style>
""", 
    unsafe_allow_html = True,
)

# Cache resource: (loaded once per Streamlit server session)

@st.cache_resource(show_spinner=' Loading configuration...')

def _load_config() -> Config:
    """Validate environment variables and return the Config singleton."""
    try:
        cfg = Config()
        return cfg
    
    except EnvironmentError as exc:
        st.error(f"Configuration Error: {exc}")
        st.info(
            "Copy `.env` to the project root and set:\n"
            "- `ANTHROPIC_API_KEY=sk-ant-…`\n"
            "- `NCBI_EMAIL=your@email.com`"
        )
        st.stop()
    
@st.cache_resource(
    show_spinner = "🧬  Loading PubMedBERT embedding model (first run takes ~30s)…"
)

def _load_pipeline(_config: Config) -> PubMedRAGPipeline:
    """
    Initialise the RAG pipeline once and cache it.

    We pass `_config` as an underscore argument to prevent Streamlit from
    hashing the Config object (which contains secrets).
    """
    try:
        pipeline = PubMedRAGPipeline(config=_config)
        return pipeline
    
    except Exception as exc:
        st.error(f"Pipeline Initialization Error: {exc}")
        st.stop()
    
# Sidebar:

def render_sidebar(config: Config) -> dict:
    """
    Render the configuration sidebar and return selected parameters as a dict.

    Returns
    -------
    dict with keys:
        num_papers, year_start, year_end,
        top_k, rerank_top_n, run_evaluation
    """

    with st.sidebar:
        st.markdown("## 🔬 PubMed RAG Research Assistant")
        st.caption("Real-time literature synthesis")
        st.divider()

        st.markdown("### Retrieval Parameters")

        num_papers = st.slider(
            "Number of papers to fetch from PubMed",
            min_value=3,
            max_value=30,
            value=config.DEFAULT_NUM_PAPERS,
            step=1,
            help=(
            "How many papers to pull from PubMed. "
            "More papers = richer context but longer index build time."
        ),
        )

    st.markdown("**Publication Year Range")
    col_a, col_b = st.columns(2)
    with col_a:
        year_start = st.number_input(
                "From", min_value=1990, max_value=2026, value=2019, step=1
            )
    with col_b:
        year_end = st.number_input(
                "To", min_value=1990, max_value=2026, value=2026, step=1
            )

    
    st.divider()

    st.markdown("### RAG Parameters")

    top_k = st.slider(
        "Number of top papers to retrieve for RAG",
        min_value=2,
        max_value=15,
        value=config.DEFAULT_TOP_K,
        step=1,
        help=(
            "How many papers to retrieve from the index for RAG. "
            "More papers = richer context but longer response time."
        ),
    )

    rerank_top_n = st.slider(
        "Rerank Top-N (FlashRank)",
            min_value=1,
            max_value=min(top_k, 5),
            value=min(config.DEFAULT_RERANK_TOP_N, top_k),
            help=(
                "Documents kept after FlashRank cross-encoder reranking. "
                "These are the final contexts passed to Claude."
        ),
    )
    st.divider()

    run_evaluation = st.checkbox(
        "🧪 Run RAG evaluation",
            value=False,
            help=(
                "Evaluate faithfulness, answer relevance, and context precision "
                "using Claude as the judge. Requires 3 additional API calls."
        ),
    )

    st.divider()

    st.markdown("### System Configuration")
    st.caption(f"LLM: `{config.LLM_MODEL}`")
    st.caption(f"Embed: `{config.EMBEDDING_MODEL.split('/')[-1]}`")
    st.caption("Reranker: `FlashRank`")
    st.caption("VectorDB: `Qdrant (in-memory)`")

    return {
        "num_papers": num_papers,
        "year_start": int(year_start),
        "year_end": int(year_end),
        "top_k": top_k,
        "rerank_top_n": rerank_top_n,
        "run_evaluation": run_evaluation,
    }

# Components Renderers:

def render_paper_card(paper: PubMedPaper, idx: int) -> None:
    """Render a single PubMed paper as a styled HTML card."""
    st.markdown(
        f"""<div class="paper-card">
<strong>{idx}. <a href="{paper.url}" target="_blank" rel="noopener">{paper.title}</a></strong><br>
<small>{paper.authors_display}</small><br>
<small><em>{paper.journal}</em> &nbsp;|&nbsp; {paper.year} &nbsp;|&nbsp; PMID&nbsp;{paper.pmid}</small>
</div>""",
        unsafe_allow_html=True,
    )


def render_eval_panel(result: EvaluationResult) -> None:
    """Render evaluation metric cards and reasoning detail."""
    st.markdown("### RAG Evaluation")

    col1, col2, col3, col4 = st.columns(4)

    def _delta_label(score: float) -> str:
        if score >= 0.75:
            return "Good"
        if score >= 0.50:
            return "Fair"
        return "Needs work"

    with col1:
        st.metric(
            "Faithfulness",
            f"{result.faithfulness:.2f}",
            _delta_label(result.faithfulness),
            help="Are all answer claims grounded in the retrieved context?",
        )
    with col2:
        st.metric(
            "Answer Relevance",
            f"{result.answer_relevance:.2f}",
            _delta_label(result.answer_relevance),
            help="Does the answer address the research question?",
        )
    with col3:
        st.metric(
            "Context Precision",
            f"{result.context_precision:.2f}",
            _delta_label(result.context_precision),
            help="Are the retrieved documents topically relevant?",
        )
    with col4:
        st.metric(
            "Overall",
            f"{result.overall_score:.2f}",
            f"Grade: {result.grade}",
            help="Weighted score (faithfulness 40%, relevance 40%, precision 20%).",
        )

    with st.expander("Evaluation reasoning"):
        st.markdown(f"**Faithfulness:** {result.faithfulness_reasoning}")
        st.markdown(f"**Answer Relevance:** {result.relevance_reasoning}")
        st.markdown(f"**Context Precision:** {result.precision_reasoning}")

        if result.unsupported_claims:
            st.markdown("**Claims not supported by context:**")
            for claim in result.unsupported_claims:
                st.markdown(f"- _{claim}_")

# main:

def main() -> None:
    """Application entry point."""

    # Header 
    st.markdown(
        """
<div class="rag-header">
    <h1>🔬 PubMed RAG Research Assistant</h1>
    <p>Real-time biomedical literature synthesis &nbsp;·&nbsp;
       PubMedBERT&nbsp;embeddings&nbsp;→&nbsp;FlashRank&nbsp;reranking&nbsp;→&nbsp;Claude&nbsp;synthesis</p>
    <p>Query expansion via MeSH terms &nbsp;·&nbsp; Strict inline citations &nbsp;·&nbsp;
       Qdrant vector search</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Load shared resources
    config = _load_config()
    pipeline = _load_pipeline(config)
    fetcher = PubMedFetcher(config=config)

    # Sidebar
    params = render_sidebar(config)

    # Search input 
    st.markdown("## 🔍 Research Question")

    query = st.text_area(
        label="Enter your biomedical research question:",
        placeholder=(
            "e.g., What are the molecular mechanisms underlying CRISPR-Cas9 "
            "off-target effects, and what strategies have been developed to "
            "minimise them for therapeutic applications?"
        ),
        height=110,
        key="query_input",
        help="Ask any biotechnology or biomedical research question.",
    )

    col_search, col_clear, _ = st.columns([2, 1, 4])
    with col_search:
        search_clicked = st.button(
            "🚀  Search & Synthesise",
            type="primary",
            use_container_width=True,
        )
    with col_clear:
        clear_clicked = st.button("🗑️  Clear", use_container_width=True)

    if clear_clicked:
        for key in ("rag_result", "rag_papers", "rag_query", "expanded_query"):
            st.session_state.pop(key, None)
        st.rerun()

    # Pipeline execution
    if search_clicked:
        _query = query.strip()
        if not _query:
            st.warning("⚠️  Please enter a research question before searching.")
            st.stop()

        if params["year_start"] > params["year_end"]:
            st.error("❌  Start year must not exceed end year.")
            st.stop()

        st.divider()

        # Step 1: MeSH query expansion
        expanded_query: str = _query
        with st.status(
            "🧠  Step 1 / 4 — Expanding query with MeSH terms…", expanded=True
        ) as step1:
            try:
                st.write(
                    "Sending your question to Claude to identify optimal "
                    "MeSH (Medical Subject Headings) terms and build a "
                    "high-recall PubMed boolean search string…"
                )
                expanded_query = fetcher.expand_query_with_mesh(_query)
                st.write(f"✅  MeSH expansion complete.")
                step1.update(
                    label="✅  Step 1 / 4 — Query expanded with MeSH terms",
                    state="complete",
                )
            except Exception as exc:
                step1.update(
                    label=f"❌  Step 1 / 4 — Query expansion error: {exc}",
                    state="error",
                )
                st.error(f"MeSH expansion failed: {exc}")
                st.stop()

        # Show expanded query (collapsible)
        with st.expander("🔍  Expanded PubMed search string", expanded=False):
            st.code(expanded_query, language="text")

        # Step 2: PubMed fetch 
        papers: list[PubMedPaper] = []
        with st.status(
            "📚  Step 2 / 4 — Fetching papers from PubMed…", expanded=True
        ) as step2:
            try:
                st.write(
                    f"Querying NCBI PubMed for up to **{params['num_papers']}** papers "
                    f"published between **{params['year_start']}** and **{params['year_end']}**…"
                )
                pmid_list = fetcher.search_pubmed(
                    query=expanded_query,
                    num_papers=params["num_papers"],
                    year_start=params["year_start"],
                    year_end=params["year_end"],
                )

                if not pmid_list:
                    step2.update(
                        label="⚠️  Step 2 / 4 — No results found",
                        state="error",
                    )
                    st.warning(
                        "PubMed returned no results for this query and year range. "
                        "Try broadening your question or adjusting the year filter."
                    )
                    st.stop()

                st.write(f"Found **{len(pmid_list)}** matching PMIDs. Fetching full records…")
                papers = fetcher.fetch_papers(pmid_list)

                if not papers:
                    step2.update(
                        label="⚠️  Step 2 / 4 — No papers with abstracts found",
                        state="error",
                    )
                    st.warning(
                        "All retrieved papers lack abstracts. "
                        "Try a different query or broader year range."
                    )
                    st.stop()

                st.write(
                    f"✅  **{len(papers)}** papers with abstracts retrieved "
                    f"({len(pmid_list) - len(papers)} skipped — no abstract)."
                )
                step2.update(
                    label=f"✅  Step 2 / 4 — {len(papers)} papers fetched from PubMed",
                    state="complete",
                )
            except RuntimeError as exc:
                step2.update(
                    label=f"❌  Step 2 / 4 — PubMed error: {exc}",
                    state="error",
                )
                st.error(str(exc))
                st.stop()

        # Step 3: Build vector index & rerank
        with st.status(
            "⚡  Step 3 / 4 — Building vector index & setting up FlashRank…",
            expanded=True,
        ) as step3:
            st.write(
                f"Embedding **{len(papers)}** abstracts with "
                f"`{config.EMBEDDING_MODEL.split('/')[-1]}` (PubMedBERT)…"
            )
            st.write(
                f"Configuring FlashRank cross-encoder to rerank top-**{params['top_k']}** "
                f"→ top-**{params['rerank_top_n']}** contexts…"
            )
            # Index is built inside pipeline.query(); we surface this step for UX clarity.
            step3.update(
                label=(
                    f"✅  Step 3 / 4 — Qdrant index configured "
                    f"(top-{params['top_k']} → FlashRank → top-{params['rerank_top_n']})"
                ),
                state="complete",
            )

        # Step 4: RAG synthesis
        rag_result: RAGResponse | None = None
        with st.status(
            "✍️  Step 4 / 4 — Synthesising answer with Claude…", expanded=True
        ) as step4:
            try:
                st.write(
                    f"Running PubMedBERT similarity search → FlashRank → "
                    f"Claude (`{config.LLM_MODEL}`) with strict citation constraints…"
                )
                rag_result = pipeline.query(
                    question=_query,
                    papers=papers,
                    top_k=params["top_k"],
                    rerank_top_n=params["rerank_top_n"],
                )
                step4.update(
                    label="✅  Step 4 / 4 — Synthesis complete!",
                    state="complete",
                )
            except (RuntimeError, ValueError) as exc:
                step4.update(
                    label=f"❌  Step 4 / 4 — Synthesis failed: {exc}",
                    state="error",
                )
                st.error(str(exc))
                st.exception(exc)
                st.stop()

        # Persist to session state so result survives reruns
        st.session_state["rag_result"] = rag_result
        st.session_state["rag_papers"] = papers
        st.session_state["rag_query"] = _query
        st.session_state["expanded_query"] = expanded_query

    # Display results (persists across reruns via session state)
    if "rag_result" in st.session_state and st.session_state["rag_result"]:
        rag_result: RAGResponse = st.session_state["rag_result"]
        papers: list[PubMedPaper] = st.session_state.get("rag_papers", [])
        stored_query: str = st.session_state.get("rag_query", "")
        params_current = params  # use sidebar params for optional evaluation

        st.divider()

        # Synthesis answer 
        st.markdown("## 📝 Synthesised Research Summary")
        st.markdown(rag_result.answer)

        # Metadata ribbon
        meta = rag_result.metadata
        st.caption(
            f"📊 Based on **{meta.get('num_input_papers', '?')}** papers &nbsp;·&nbsp; "
            f"Retrieved top-**{meta.get('top_k', '?')}** &nbsp;·&nbsp; "
            f"Reranked to top-**{meta.get('rerank_top_n', '?')}** &nbsp;·&nbsp; "
            f"Model: `{meta.get('llm_model', '?')}`"
        )

        st.divider()

        # Source papers expander
        with st.expander(
            f"📚  Analysed papers — {len(papers)} retrieved from PubMed",
            expanded=False,
        ):
            st.markdown(
                "*All papers retrieved and indexed for this synthesis. "
                "Click any title to open the full record on PubMed.*"
            )
            for i, paper in enumerate(papers, start=1):
                render_paper_card(paper, i)

        # Optional evaluation panel
        if params_current.get("run_evaluation") and papers:
            st.divider()
            with st.status("🧪  Running RAG evaluation…", expanded=True) as eval_status:
                try:
                    evaluator = RAGEvaluator(config=config)
                    contexts = [
                        f"Title: {p.title}\n\nAbstract: {p.abstract}"
                        for p in papers[:5]
                    ]
                    eval_result = evaluator.evaluate(
                        question=stored_query,
                        answer=rag_result.answer,
                        contexts=contexts,
                    )
                    eval_status.update(
                        label="✅  Evaluation complete!", state="complete"
                    )
                    render_eval_panel(eval_result)
                except Exception as exc:
                    eval_status.update(
                        label=f"❌  Evaluation failed: {exc}", state="error"
                    )
                    st.error(f"Evaluation error: {exc}")

    # Footer
    st.divider()
    st.caption(
        "🔬 PubMed RAG Research Assistant &nbsp;·&nbsp; "
        "Data sourced from NCBI PubMed (real-time) &nbsp;·&nbsp; "
        "**Not for clinical decision-making** &nbsp;·&nbsp; "
        "Always verify findings with the primary literature."
    )


if __name__ == "__main__":
    main()
