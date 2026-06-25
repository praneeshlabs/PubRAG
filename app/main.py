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

