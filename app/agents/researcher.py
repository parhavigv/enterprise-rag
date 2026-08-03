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
    "You are a senior business analyst's research assistant. Answer the "
    "user's question using ONLY the provided context and cite the exact "
    "source numbers inline as [n] after each claim. Never invent facts, "
    "figures, or experience that is not present in the context; if the "
    "context lacks the information, say so explicitly and do not "
    "speculate.\n\n"
    "Give thorough, business-useful answers with explanations: state the "
    "bottom line up front, then support it with concrete specifics and "
    "reasoning drawn from the sources.\n\n"
    "When the user asks you to review, evaluate, critique, or improve a "
    "document (such as a resume, proposal, or report), structure your "
    "answer with plain-text section headings:\n"
    " 1. Overview - what the document contains and its purpose.\n"
    " 2. Strengths - what works well, tied to specifics in the document.\n"
    " 3. Gaps and risks - missing skills, weak areas, or concerns.\n"
    " 4. Recommendations - concrete, actionable improvements.\n"
    " 5. Bottom line - a one-sentence verdict.\n"
    "Keep every section evidence-based and focused; never pad."
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

    async def generate(
        self,
        query: str,
        documents: list[RetrievedDocument],
        images: list[str] | None = None,
    ) -> ResearchResponse:
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
            "Answer using only the context above, citing sources as [n]. "
            "If the question asks for a review or assessment, follow the "
            "structured review format in your system instructions."
        )

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._user_content(user_prompt, images)},
            ]
            result = await self._llm.complete(messages)
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
    def _user_content(prompt: str, images: list[str] | None) -> str | list[dict]:
        """Plain string prompt, or a multimodal block list when images exist."""
        if not images:
            return prompt
        blocks: list[dict] = [{"type": "text", "text": prompt}]
        for image in images:
            blocks.append(
                {"type": "image_url", "image_url": {"url": ResearcherAgent._image_data_uri(image)}}
            )
        return blocks

    @staticmethod
    def _image_data_uri(data: str) -> str:
        """Normalise raw base64 / bare 'base64,' input into a data URI."""
        data = data.strip()
        if data.startswith("data:image"):
            return data
        if data.startswith("base64,"):
            return "data:image/png;base64," + data[len("base64,") :]
        return "data:image/png;base64," + data

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
