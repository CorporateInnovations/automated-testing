import json
import logging
from typing import Any

from fastapi import HTTPException
from openai import OpenAI

from app.config import Settings
from app.models import TestPlanResponse
from app.services.chunking import chunk_text
from app.services.prompt import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger("app.llm")


class TestPlanExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.openai_api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=settings.openai_api_key)

    def _call_llm(self, user_content: str) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content or ""

    def extract(self, text: str) -> TestPlanResponse:
        max_chars = getattr(self.settings, "max_chars", 120_000)
        if not text or len(text) > max_chars:
            raise HTTPException(status_code=400, detail="Document empty or too large")

        chunk_size = getattr(self.settings, "chunk_size", 6_000)
        chunk_overlap = getattr(self.settings, "chunk_overlap", 500)
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        user_prompt = build_user_prompt(chunks or [text])

        last_err: Any = None
        max_retries = getattr(self.settings, "max_retries", 2)
        for _ in range(max_retries + 1):
            try:
                logger.info(
                    "Calling LLM: model=%s chunks=%s prompt_chars=%s",
                    self.settings.model,
                    len(chunks),
                    len(user_prompt),
                )
                raw = self._call_llm(user_prompt)
                data = self._normalize_payload(json.loads(raw))
                return TestPlanResponse(**data)
            except Exception as exc:
                logger.exception("LLM attempt failed to return valid JSON")
                last_err = exc
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {last_err}")

    def _normalize_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        """Coerce/patch missing fields so Pydantic validation does not choke on LLM quirks."""
        stories = data.get("user_stories", []) or []
        normalized_stories = []
        for idx, story in enumerate(stories):
            s = dict(story)
            s.setdefault("id", f"US-{idx+1:03d}")
            s.setdefault("feature", "unspecified")
            s.setdefault("title", "")
            s.setdefault("role", "")
            s.setdefault("goal", "")
            s.setdefault("benefit", "")
            if isinstance(s.get("acceptance_criteria"), str):
                s["acceptance_criteria"] = [s["acceptance_criteria"]]
            s.setdefault("acceptance_criteria", [])
            if isinstance(s.get("related_test_ids"), str):
                s["related_test_ids"] = [s["related_test_ids"]]
            s.setdefault("related_test_ids", [])
            normalized_stories.append(s)
        data["user_stories"] = normalized_stories

        cases = data.get("test_cases", []) or []
        normalized_cases = []
        for idx, case in enumerate(cases):
            c = dict(case)
            c.setdefault("id", f"TC-{idx+1:03d}")
            c.setdefault("feature", "unspecified")
            c.setdefault("title", "")
            c.setdefault("steps", [])
            c.setdefault("expected_result", "")
            c.setdefault("priority", "P2")
            c.setdefault("type", "functional")
            c.setdefault("tags", [])

            if isinstance(c["steps"], str):
                c["steps"] = [c["steps"]]
            if c["steps"] is None:
                c["steps"] = []

            if isinstance(c["tags"], str):
                c["tags"] = [c["tags"]]
            if c["tags"] is None:
                c["tags"] = []

            trace = c.get("trace")
            if isinstance(trace, list):
                c["trace"] = "; ".join(str(t) for t in trace)
            elif trace is None:
                c["trace"] = None

            normalized_cases.append(c)

        data["test_cases"] = normalized_cases
        data.setdefault("assumptions", [])
        data.setdefault("risks", [])
        return data
