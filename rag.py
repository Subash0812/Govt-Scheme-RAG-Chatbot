"""Small, transparent retrieval-augmented generation layer for SchemeSathi."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RetrievedScheme:
    record: dict
    score: float


@dataclass
class Answer:
    answer: str
    sources: list[dict]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class SchemeRetriever:
    def __init__(self, schemes: list[dict]) -> None:
        self.schemes = schemes
        self.documents = [self._to_document(scheme) for scheme in schemes]
        self.tokens = [tokenize(document) for document in self.documents]

    @classmethod
    def from_json(cls, path: Path) -> "SchemeRetriever":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _to_document(scheme: dict) -> str:
        fields = [scheme["name"], scheme["summary"], scheme["benefits"], scheme["eligibility"], scheme["how_to_apply"], " ".join(scheme["keywords"])]
        return " ".join(fields)

    def retrieve(self, query: str, limit: int = 3) -> list[RetrievedScheme]:
        query_tokens = tokenize(query)
        # Lightweight term-frequency scoring keeps the demo fully local and transparent.
        scores = [sum(Counter(tokens)[term] for term in query_tokens) for tokens in self.tokens]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:limit]
        return [RetrievedScheme(self.schemes[index], float(score)) for index, score in ranked if score > 0]


def _sources(matches: list[RetrievedScheme]) -> list[dict]:
    seen: set[str] = set()
    sources = []
    for match in matches:
        scheme = match.record
        if scheme["name"] not in seen:
            sources.append({"title": scheme["name"], "url": scheme["official_url"]})
            seen.add(scheme["name"])
    return sources


def _context(matches: list[RetrievedScheme]) -> str:
    return "\n\n".join(
        f"SCHEME: {m.record['name']}\nSUMMARY: {m.record['summary']}\nBENEFITS: {m.record['benefits']}\nELIGIBILITY: {m.record['eligibility']}\nAPPLY: {m.record['how_to_apply']}"
        for m in matches
    )


def _local_answer(matches: list[RetrievedScheme]) -> str:
    if not matches:
        return ("I could not find a reliable match in this demo knowledge base. Try naming a scheme or browse "
                "[myScheme](https://www.myscheme.gov.in/) for official, up-to-date options.")
    lines = ["Here are the most relevant schemes from the knowledge base:"]
    for match in matches[:2]:
        s = match.record
        lines.append(f"\n**{s['name']}** — {s['summary']}\n\n**Benefits:** {s['benefits']}\n\n**Broad eligibility:** {s['eligibility']}\n\n**How to apply:** {s['how_to_apply']}")
    lines.append("\nPlease check the linked official portal before applying, as rules and benefits can change.")
    return "\n".join(lines)


def answer_question(question: str, retriever: SchemeRetriever) -> Answer:
    matches = retriever.retrieve(question)
    sources = _sources(matches)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key and matches:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = (
                "You are SchemeSathi, a helpful assistant for Indian government schemes. "
                "Answer only using the supplied context. Be concise, use plain English, "
                "say when information is not available, and never claim someone is definitely eligible. "
                "End by asking the user to verify on the official portal.\n\n"
                f"Question: {question}\n\nRetrieved context:\n{_context(matches)}"
            )
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt
            )
            return Answer(response.text or _local_answer(matches), sources)
        except Exception:
            pass  # The app stays useful without an API key or network connection.
    return Answer(_local_answer(matches), sources)
