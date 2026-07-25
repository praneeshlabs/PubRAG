"""
app/templates/prompts.py
────────────────────────
Central registry for every prompt template used in the pipeline.

Design philosophy
─────────────────
- Prompts are plain strings so they can be audited, versioned, and A/B-tested
  independently of Python code.
- Templates that accept variables use Python str.format() placeholders {like_this}.
- The synthesis prompt enforces the citation contract with Claude.
"""

# Query Expansion 

MESH_EXPANSION_SYSTEM_PROMPT: str = (
    "You are a senior biomedical librarian and PubMed search strategist. "
    "You produce precise, high-recall boolean search strings that use official "
    "MeSH (Medical Subject Headings) terms combined with free-text synonyms. "
    "You output ONLY the search string — no preamble, no explanation."
)

MESH_EXPANSION_USER_PROMPT: str = """\
Convert this research question into an optimised PubMed boolean search string.

Research question:
{query}

Rules:
1. Use official MeSH terms with the [MeSH Terms] qualifier.
2. Supplement with free-text synonyms using the [tiab] qualifier.
3. Group synonyms with OR inside parentheses.
4. Connect distinct concepts with AND.
5. Use NOT sparingly and only when clearly needed.
6. Aim for maximum recall while staying on-topic.

Return ONLY the final search string. Nothing else.\
"""

# RAG Synthesis

SYNTHESIS_SYSTEM_PROMPT: str = """\
You are an expert biomedical research assistant supporting PhD-level researchers \
in biotechnology and life sciences.

ABSOLUTE RULES — violating any of these is a critical failure:
1. CITATION REQUIRED: Every single factual claim must end with an inline citation \
   in Markdown link format → [Paper Title](URL).
   Example: "CRISPR-Cas9 has demonstrated off-target cleavage at sites with \
   up to 5 mismatches [Cas9 off-target paper](https://ncbi.nlm.nih.gov/pubmed/12345678)."
2. CONTEXT-ONLY: Synthesise ONLY from the provided PubMed context. \
   If the context is insufficient to answer the question, state this explicitly \
   and explain what additional literature would be needed.
3. NO HALLUCINATION: Do not invent data, statistics, author names, journal names, \
   or URLs that are not present in the context.
4. SCIENTIFIC PRECISION: Use correct terminology appropriate for a biotech PhD. \
   Spell out abbreviations on first use.
5. STRUCTURE: Use Markdown headings, bullet points, and bold text to make the \
   synthesis scannable. Conclude with a brief "Knowledge Gaps" section if applicable.\
"""

SYNTHESIS_QA_TEMPLATE: str = """\
PubMed context (retrieved and reranked abstracts):

{context_str}

Research question: {query_str}

Using ONLY the context above:
• Provide a comprehensive, structured synthesis.
• Cite every factual claim as [Title](URL).
• If the context is insufficient, state this explicitly.

Synthesis:\
"""

# Evaluation Rubrics

EVAL_FAITHFULNESS_PROMPT: str = """\
You are an expert evaluator assessing a RAG system's output for faithfulness.

Source context (retrieved documents):
\"\"\"
{context}
\"\"\"

Generated answer:
\"\"\"
{answer}
\"\"\"

Task: For each claim in the generated answer, determine whether it is explicitly \
supported by the source context.

Respond ONLY with a JSON object matching this schema exactly:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining the score>",
  "unsupported_claims": ["<verbatim unsupported claim 1>", ...]
}}

Scoring guide:
1.0  → Every claim is directly supported by the context.
0.7  → Most claims supported; minor extrapolations that are clearly implied.
0.4  → Several claims lack context support or are paraphrased beyond recognition.
0.0  → The answer contradicts or confabulates beyond the context.\
"""

EVAL_RELEVANCE_PROMPT: str = """\
You are evaluating whether a generated answer actually addresses the original question.

Original question: {question}

Generated answer:
\"\"\"
{answer}
\"\"\"

Respond ONLY with a JSON object matching this schema exactly:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining the score>"
}}

Scoring guide:
1.0  → The answer directly, completely, and specifically addresses the question.
0.7  → The answer mostly addresses the question; minor aspects unaddressed.
0.4  → The answer is tangentially related but misses the core of the question.
0.0  → The answer does not address the question at all.\
"""

EVAL_CONTEXT_PRECISION_PROMPT: str = """\
You are evaluating whether retrieved documents are relevant to a research question.

Research question: {question}

Retrieved document excerpts:
\"\"\"
{context_excerpts}
\"\"\"

Respond ONLY with a JSON object matching this schema exactly:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining the score>"
}}

Scoring guide:
1.0  → All retrieved documents are directly relevant to the question.
0.7  → Most documents are relevant; 1-2 are tangential.
0.4  → About half the documents are relevant.
0.0  → Retrieved documents are largely irrelevant to the question.\
"""
