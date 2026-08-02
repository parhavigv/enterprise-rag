"""ResearcherAgent - grounded answer generation over retrieved documents.

Pipeline:
    1. Format retrieved documents into a numbered context block.
    2. Ask the LLM to answer *only* from the provided context, citing
       source numbers inline, or to say the answer is not in the context.
    3. Return a structured :class:`ResearchResponse` with the answer,
       sources used, and generation metadata.

Graceful degradation: if the LLM is unavailable (or no documents were
retrieved) the agent falls back to an extractive summary so the API still
returns a useful response instead of an error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.agents.llm_client import AsyncLLMClient, LLMUnavailableError
from app.core.logging import get_logger
from retrieval.types import RetrievedDocument

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using "
    "ONLY the provided context. Cite sources inline as [n] where n is the "
    "source number. If the context does not contain the answer, say "
    '"The context does not contain this information" and do not invent '
    "details. Keep the answer concise and factual."
)


@dataclass(slots=True)
class ResearchResponse:
    query: str
    answer: str
    sources: list[RetrievedDocument] = field(default_factory=list)
    model: str = "fallback-extractive"
    generated: bool = False
    latency_ms: float = 0.0
    error: str | None = None


class ResearcherAgent:
    def __init__(self, llm: AsyncLLMClient, max_context_docs: int = 5) -> None:
        self._llm = llm
        self._max_context_docs = max_context_docs

    @property
    def model(self) -> str:
        return self._llm.model

    async def generate(self, query: str, documents: list[RetrievedDocument]) -> ResearchResponse:
        t0 = time.perf_counter()
        sources = documents[: self._max_context_docs]

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # Extractive fallback when there is nothing to ground on.
        if not sources:
            return ResearchResponse(
                query=query,
                answer="No relevant documents were found in the index.",
                sources=[],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        context = self._format_context(sources)
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above, citing sources as [n]."
        )

        try:
            result = await self._llm.complete(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            answer = result.text
            logger.info(
                "LLM generation ok | model={} | tokens={} | latency={:.0f}ms",
                result.model,
                result.total_tokens,
                result.latency_ms,
            )
            return ResearchResponse(
                query=query,
                answer=answer,
                sources=sources,
                model=result.model,
                generated=True,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except LLMUnavailableError as e:
            logger.warning("LLM unavailable - using extractive fallback: {}", e)
            return ResearchResponse(
                query=query,
                answer=self._extractive_summary(sources),
                sources=sources,
                model="fallback-extractive",
                generated=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=str(e),
            )

    # ------------------------------------------------------------------ #
    # Prompt / fallback internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_context(documents: list[RetrievedDocument]) -> str:
        blocks = []
        for i, doc in enumerate(documents, start=1):
            meta = doc.metadata or {}
            header = f"[{i}] source={meta.get('source', 'unknown')}"
            if meta.get("page_number"):
                header += f", page={meta['page_number']}"
            blocks.append(f"{header}\n{doc.text.strip()}")
        return "\n\n".join(blocks)

    @staticmethod
    def _extractive_summary(documents: list[RetrievedDocument]) -> str:
        """Fallback: return the strongest passages with their sources."""
        parts = []
        for i, doc in enumerate(documents[:3], start=1):
            snippet = doc.text.strip()
            if len(snippet) > 600:
                snippet = snippet[:600].rsplit(" ", 1)[0] + "..."
            meta = doc.metadata or {}
            src = meta.get("source", "unknown")
            parts.append(f"[{i}] ({src}) {snippet}")
        return "\n\n".join(parts)
