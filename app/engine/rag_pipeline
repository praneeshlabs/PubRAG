"""
app/engine/rag_pipeline.py
──────────────────────────
Hybrid RAG pipeline for PubMed literature synthesis.

Architecture
────────────
  PubMedPaper list
        │
        ▼
  LlamaIndex Document nodes  (text = title + authors + abstract, metadata = PMID/URL/…)
        │
        ▼
  NeuML/pubmedbert-base-embeddings  (domain-adapted sentence transformer)
        │
        ▼
  Qdrant in-memory vector store     (cosine similarity search, top-K candidates)
        │
        ▼
  FlashRank cross-encoder reranker  (re-scores top-K → top-N)
        │
        ▼
  Claude (claude-opus-4-6)         (strict citation synthesis)
        │
        ▼
  Structured Markdown answer

Design notes
────────────
- LlamaIndex global `Settings` are configured once per pipeline instance.
  Streamlit's @st.cache_resource ensures a single instance per session.
- A fresh Qdrant `:memory:` collection is built for every new set of papers.
  This is intentional: PubMed results differ per query.
- FlashRankRerank uses a lightweight cross-encoder that runs CPU-only.
"""
from __future__ import annotations

import logging
from typing import Optional

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic as AnthropicLLM
from llama_index.postprocessor.flashrank_rerank import FlashRankRerank
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.core.config import Config
from app.core.pubmed_api import PubMedPaper
from app.templates.prompts import SYNTHESIS_QA_TEMPLATE, SYNTHESIS_SYSTEM_PROMPT

logger = logging.getLogger(__name__)



# Response model

class RAGResponse:
    """
    Container for a completed RAG query result.

    Attributes
    ----------
    answer        : The final synthesised Markdown answer from Claude.
    source_nodes  : Reranked LlamaIndex NodeWithScore objects (for attribution UI).
    expanded_query: The MeSH-expanded search string (for display).
    metadata      : Diagnostic information (retrieval counts, model names, …).
    """

    def __init__(
        self,
        answer: str,
        source_nodes: list[NodeWithScore],
        expanded_query: str = "",
        metadata: Optional[dict] = None,
    ) -> None:
        self.answer = answer
        self.source_nodes = source_nodes
        self.expanded_query = expanded_query
        self.metadata = metadata or {}

    def get_context_strings(self) -> list[str]:
        """Return plain-text context from each source node (for evaluation)."""
        return [node.node.get_content() for node in self.source_nodes]


# Pipeline
class PubMedRAGPipeline:
    """
    End-to-end RAG pipeline: embeddings → vector search → rerank → synthesise.

    Parameters
    ----------
    config : Config
        Validated application configuration.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._configure_llama_index_settings()

    # Initialisation 

    def _configure_llama_index_settings(self) -> None:
        """
        Initialise LlamaIndex global Settings once per pipeline instance.

        Global Settings are the idiomatic way to configure the embedding model
        and LLM in LlamaIndex v0.10+.  We do this once here so every index and
        query engine created by this class shares the same models without
        repeatedly loading weights.
        """
        logger.info("Loading PubMedBERT embedding model: %s", self.config.EMBEDDING_MODEL)
        try:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=self.config.EMBEDDING_MODEL,
                trust_remote_code=True,
                embed_batch_size=32,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{self.config.EMBEDDING_MODEL}'. "
                "Ensure sentence-transformers and torch are installed. "
                f"Original error: {exc}"
            ) from exc

        logger.info("Initialising Claude LLM: %s", self.config.LLM_MODEL)
        try:
            Settings.llm = AnthropicLLM(
                model=self.config.LLM_MODEL,
                api_key=self.config.ANTHROPIC_API_KEY,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                max_tokens=self.config.MAX_TOKENS,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise Anthropic LLM '{self.config.LLM_MODEL}'. "
                f"Check ANTHROPIC_API_KEY and llama-index-llms-anthropic version. "
                f"Original error: {exc}"
            ) from exc

        Settings.chunk_size = self.config.CHUNK_SIZE
        Settings.chunk_overlap = self.config.CHUNK_OVERLAP
        logger.info(
            "LlamaIndex Settings configured | chunk_size=%d | chunk_overlap=%d",
            self.config.CHUNK_SIZE,
            self.config.CHUNK_OVERLAP,
        )

    # Document preparation

    def _papers_to_documents(self, papers: list[PubMedPaper]) -> list[Document]:
        """
        Convert PubMedPaper objects to LlamaIndex Document nodes.

        The `text` field is what gets embedded.  We structure it to give the
        embedding model enough context: title (most important for similarity),
        followed by authors/journal (useful for scoping), then the abstract body.

        Critical metadata stored explicitly:
            pmid, title, year, url — used for inline citations.
        """
        documents: list[Document] = []

        for paper in papers:
            # Build a semantically rich text block for embedding
            text_body = (
                f"Title: {paper.title}\n\n"
                f"Authors: {paper.authors_display}\n"
                f"Journal: {paper.journal} ({paper.year})\n"
                f"PMID: {paper.pmid}\n\n"
                f"Abstract:\n{paper.abstract}"
            )

            doc = Document(
                text=text_body,
                metadata={
                    "pmid": paper.pmid,
                    "title": paper.title,
                    "authors": paper.authors_display,
                    "journal": paper.journal,
                    "year": paper.year,
                    "url": paper.url,
                },
                # Prevent metadata keys from polluting the LLM context window
                # but keep them available for citation extraction.
                excluded_llm_metadata_keys=["pmid", "authors", "journal", "year"],
                metadata_separator="\n",
                metadata_template="{key}: {value}",
                text_template="{content}",
            )
            documents.append(doc)
            logger.debug("Document created for PMID %s (%s)", paper.pmid, paper.year)

        logger.info(
            "Converted %d PubMedPaper objects to LlamaIndex Documents.", len(documents)
        )
        return documents

    # Index construction

    def _build_qdrant_index(self, documents: list[Document]) -> VectorStoreIndex:
        """
        Embed documents with PubMedBERT and store them in an in-memory Qdrant
        collection.

        A fresh collection is created for each call.  This keeps each search
        isolated so papers from different queries don't contaminate each other.

        Parameters
        ----------
        documents : list[Document]
            Pre-processed LlamaIndex Document nodes.

        Returns
        -------
        VectorStoreIndex
            Ready-to-query index backed by Qdrant.
        """
        logger.info(
            "Building Qdrant in-memory index for %d documents…", len(documents)
        )
        try:
            # `:memory:` creates a volatile, in-process Qdrant instance — no
            # disk I/O, no server required.  Perfect for per-query indices.
            qdrant_client = QdrantClient(":memory:")

            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=self.config.COLLECTION_NAME,
            )
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )

            index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                show_progress=False,
            )

            logger.info("Qdrant index built successfully.")
            return index

        except Exception as exc:
            raise RuntimeError(
                f"Failed to build Qdrant vector index: {exc}. "
                "Check qdrant-client and llama-index-vector-stores-qdrant versions."
            ) from exc

    # Query engine construction

    def _build_query_engine(
        self,
        index: VectorStoreIndex,
        top_k: int,
        rerank_top_n: int,
    ) -> RetrieverQueryEngine:
        """
        Compose a RetrieverQueryEngine with:
            • VectorIndexRetriever (PubMedBERT similarity search)
            • FlashRankRerank (cross-encoder reranking)
            • ResponseSynthesizer (Claude with strict citation prompt)

        Parameters
        ----------
        index        : Built VectorStoreIndex.
        top_k        : Initial retrieval count before reranking.
        rerank_top_n : Final context count after reranking (passed to Claude).
        """
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
        )

        try:
            reranker = FlashRankRerank(top_n=rerank_top_n)
        except Exception as exc:
            logger.warning(
                "FlashRankRerank initialisation failed (%s). "
                "Proceeding without reranking. "
                "Install flashrank: pip install flashrank",
                exc,
            )
            reranker = None  # type: ignore[assignment]

        synthesis_prompt = PromptTemplate(SYNTHESIS_QA_TEMPLATE)
        response_synthesizer = get_response_synthesizer(
            text_qa_template=synthesis_prompt,
            response_mode="compact",
            use_async=False,
        )

        postprocessors = [reranker] if reranker is not None else []

        engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors,
        )
        return engine

    # Public interface

    def query(
        self,
        question: str,
        papers: list[PubMedPaper],
        top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ) -> RAGResponse:
        """
        Run the full RAG query for a given research question and paper set.

        Parameters
        ----------
        question     : The user's research question.
        papers       : PubMed papers retrieved by PubMedFetcher.
        top_k        : Candidates to retrieve from the vector store.
                       Defaults to config.DEFAULT_TOP_K.
        rerank_top_n : Documents kept after FlashRank.
                       Defaults to config.DEFAULT_RERANK_TOP_N.

        Returns
        -------
        RAGResponse
            Contains the synthesised answer, source nodes, and diagnostics.

        Raises
        ------
        ValueError    : If papers list is empty.
        RuntimeError  : If any pipeline stage fails unrecoverably.
        """
        if not papers:
            raise ValueError(
                "No papers provided to the RAG pipeline. "
                "Ensure PubMed returned results before calling query()."
            )

        _top_k = top_k if top_k is not None else self.config.DEFAULT_TOP_K
        _rerank_top_n = (
            rerank_top_n
            if rerank_top_n is not None
            else self.config.DEFAULT_RERANK_TOP_N
        )

        # Clamp so we never ask for more than we have
        _top_k = min(_top_k, len(papers))
        _rerank_top_n = min(_rerank_top_n, _top_k)

        logger.info(
            "RAG query started | papers=%d | top_k=%d | rerank_top_n=%d",
            len(papers),
            _top_k,
            _rerank_top_n,
        )

        # 1. Prepare documents
        documents = self._papers_to_documents(papers)

        # 2. Build vector index
        index = self._build_qdrant_index(documents)

        # 3. Assemble query engine
        engine = self._build_query_engine(
            index=index,
            top_k=_top_k,
            rerank_top_n=_rerank_top_n,
        )

        # 4. Execute query
        logger.info("Executing RAG query: %.120s…", question)
        try:
            llama_response = engine.query(question)
        except Exception as exc:
            raise RuntimeError(
                f"RAG query execution failed: {exc}. "
                "Check Anthropic API key, model name, and network connectivity."
            ) from exc

        answer_text = str(llama_response).strip()
        source_nodes: list[NodeWithScore] = llama_response.source_nodes or []

        logger.info(
            "RAG query complete | answer_len=%d chars | source_nodes=%d",
            len(answer_text),
            len(source_nodes),
        )

        return RAGResponse(
            answer=answer_text,
            source_nodes=source_nodes,
            metadata={
                "num_input_papers": len(papers),
                "top_k": _top_k,
                "rerank_top_n": _rerank_top_n,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "llm_model": self.config.LLM_MODEL,
            },
        )
