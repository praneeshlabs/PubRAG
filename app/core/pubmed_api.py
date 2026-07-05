"""
app/core/pubmed_api.py
──────────────────────
PubMed data-fetching module.

Pipeline
────────
1. Claude expands the raw user question into a MeSH boolean search string.
2. Bio.Entrez.esearch() retrieves matching PMIDs.
3. Bio.Entrez.efetch() downloads full Medline records.
4. Records are parsed into typed PubMedPaper dataclasses.

Design notes
────────────
- Every public method has a try/except with descriptive re-raises so the UI
  can surface meaningful messages without exposing raw stack traces.
- Entrez calls are separated so the UI can show incremental progress.
- Papers without abstracts are silently skipped (useless for RAG).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import anthropic
from Bio import Entrez, Medline

from app.core.config import Config
from app.templates.prompts import (
    MESH_EXPANSION_SYSTEM_PROMPT,
    MESH_EXPANSION_USER_PROMPT,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PubMedPaper:
    """
    Typed representation of a single PubMed record.

    All fields have safe defaults so downstream code never encounters
    AttributeError on partially-populated records.
    """

    pmid: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    journal: str = "Unknown Journal"
    year: str = "Unknown"
    url: str = ""

    def __post_init__(self) -> None:
        if not self.url and self.pmid:
            self.url = f"https://ncbi.nlm.nih.gov/pubmed/{self.pmid}"

    # ── Convenience helpers ───────────────────────────────────────────────

    @property
    def authors_display(self) -> str:
        """Return a human-readable author string capped at 5 names."""
        if not self.authors:
            return "Unknown Authors"
        names = self.authors[:5]
        suffix = " et al." if len(self.authors) > 5 else ""
        return ", ".join(names) + suffix

    def to_dict(self) -> dict:
        """Serialise to plain dict for JSON logging / persistence."""
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "url": self.url,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fetcher
# ─────────────────────────────────────────────────────────────────────────────


class PubMedFetcher:
    """
    Orchestrates query expansion → NCBI search → record fetching.

    Parameters
    ----------
    config : Config
        Application configuration (API keys, email, defaults).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        # NCBI requires a registered email to comply with their usage policy.
        Entrez.email = config.NCBI_EMAIL
        logger.info("PubMedFetcher initialised (NCBI email: %s)", config.NCBI_EMAIL)

    # ── Public API ─────────────────────────────────────────────────────────

    def expand_query_with_mesh(self, raw_query: str) -> str:
        """
        Use Claude to convert a natural-language research question into an
        optimal PubMed boolean search string using MeSH terms.

        Parameters
        ----------
        raw_query : str
            The user's research question as typed.

        Returns
        -------
        str
            A PubMed-compatible boolean search string.
            Falls back to the raw query if the LLM call fails.
        """
        prompt = MESH_EXPANSION_USER_PROMPT.format(query=raw_query)

        try:
            logger.info("Calling Claude for MeSH query expansion...")
            response = self._anthropic_client.messages.create(
                model=self.config.LLM_MODEL,
                max_tokens=512,
                system=MESH_EXPANSION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            expanded: str = response.content[0].text.strip()
            # Safety: strip markdown code fences if Claude wraps the output
            expanded = expanded.strip("`").strip()
            logger.info("Expanded query: %s", expanded[:200])
            return expanded

        except anthropic.APIStatusError as exc:
            logger.warning(
                "Anthropic API error during query expansion (status %s): %s. "
                "Falling back to raw query.",
                exc.status_code,
                exc.message,
            )
            return raw_query

        except anthropic.APIConnectionError as exc:
            logger.warning(
                "Connection error during query expansion: %s. Falling back.", exc
            )
            return raw_query

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Unexpected error during query expansion: %s. Falling back.", exc
            )
            return raw_query

    def search_pubmed(
        self,
        query: str,
        num_papers: int = 10,
        year_start: int = 2019,
        year_end: int = 2025,
    ) -> list[str]:
        """
        Run an Entrez esearch and return a list of PubMed IDs.

        Parameters
        ----------
        query       : PubMed search string (MeSH boolean or free text).
        num_papers  : Maximum number of PMIDs to return.
        year_start  : Earliest publication year (inclusive).
        year_end    : Latest publication year (inclusive).

        Returns
        -------
        list[str]
            Ordered list of PMIDs, most relevant first.

        Raises
        ------
        RuntimeError
            If the Entrez search call itself fails.
        """
        logger.info(
            "Searching PubMed | papers=%d | years=%d-%d | query=%.120s…",
            num_papers,
            year_start,
            year_end,
            query,
        )
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=num_papers,
                sort="relevance",
                mindate=str(year_start),
                maxdate=str(year_end),
                datetype="pdat",
            )
            record = Entrez.read(handle)
            handle.close()

            pmids: list[str] = record.get("IdList", [])
            logger.info("PubMed search returned %d PMIDs.", len(pmids))
            return pmids

        except Exception as exc:
            raise RuntimeError(
                f"PubMed esearch failed: {exc}. "
                "Check your internet connection and NCBI_EMAIL setting."
            ) from exc

    def fetch_papers(self, pmid_list: list[str]) -> list[PubMedPaper]:
        """
        Fetch full Medline records for the given PMIDs and parse them into
        PubMedPaper objects.

        Papers without abstracts are excluded because they provide no useful
        context for the RAG pipeline.

        Parameters
        ----------
        pmid_list : list[str]
            List of PubMed IDs to fetch.

        Returns
        -------
        list[PubMedPaper]
            Parsed paper objects that have non-empty abstracts.

        Raises
        ------
        RuntimeError
            If the Entrez efetch call fails.
        """
        if not pmid_list:
            logger.warning("fetch_papers called with empty pmid_list.")
            return []

        logger.info("Fetching Medline records for %d PMIDs…", len(pmid_list))
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmid_list),
                rettype="medline",
                retmode="text",
            )
            records = list(Medline.parse(handle))
            handle.close()
        except Exception as exc:
            raise RuntimeError(
                f"PubMed efetch failed for {len(pmid_list)} IDs: {exc}"
            ) from exc

        papers: list[PubMedPaper] = []
        skipped_no_abstract = 0

        for record in records:
            pmid = str(record.get("PMID", "")).strip()
            abstract = record.get("AB", "").strip()

            if not abstract:
                skipped_no_abstract += 1
                logger.debug("PMID %s skipped: no abstract.", pmid)
                continue

            title = record.get("TI", "No title available").strip()
            # Authors stored as a list in Medline format
            raw_authors = record.get("AU", [])
            authors: list[str] = (
                list(raw_authors) if isinstance(raw_authors, list) else [raw_authors]
            )
            # Journal — prefer full name, fall back to abbreviation
            journal = record.get("JT", record.get("TA", "Unknown Journal")).strip()
            # Date published — first 4 chars are the year
            date_pub: str = record.get("DP", "")
            year = date_pub[:4] if date_pub and date_pub[:4].isdigit() else "Unknown"

            paper = PubMedPaper(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
            )
            papers.append(paper)

        logger.info(
            "Parsed %d papers (%d skipped — no abstract).",
            len(papers),
            skipped_no_abstract,
        )
        return papers

    def fetch(
        self,
        raw_query: str,
        num_papers: int = 10,
        year_start: int = 2019,
        year_end: int = 2025,
    ) -> tuple[str, list[PubMedPaper]]:
        """
        Full convenience pipeline:
            raw_query → MeSH expansion → PubMed search → paper fetch.

        Parameters
        ----------
        raw_query   : Natural-language research question.
        num_papers  : Papers to retrieve from PubMed.
        year_start  : Earliest publication year.
        year_end    : Latest publication year.

        Returns
        -------
        tuple[str, list[PubMedPaper]]
            (expanded_query_string, list_of_papers)
        """
        expanded_query = self.expand_query_with_mesh(raw_query)
        # Brief pause to respect Entrez rate limits between calls
        time.sleep(self.config.ENTREZ_RATE_LIMIT_SLEEP)

        pmid_list = self.search_pubmed(
            query=expanded_query,
            num_papers=num_papers,
            year_start=year_start,
            year_end=year_end,
        )
        time.sleep(self.config.ENTREZ_RATE_LIMIT_SLEEP)

        papers = self.fetch_papers(pmid_list)
        return expanded_query, papers
