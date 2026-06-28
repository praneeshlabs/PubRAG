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
    show spinner = "🧬  Loading PubMedBERT embedding model (first run takes ~30s)…"
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
            help=
            "How many papers to pull from PubMed. "
            "More papers = richer context but longer index build time."
        )

        