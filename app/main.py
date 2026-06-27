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
