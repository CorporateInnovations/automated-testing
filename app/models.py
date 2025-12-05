from typing import List, Optional

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    id: str = Field(..., example="TC-001")
    feature: str
    title: str
    steps: List[str]
    expected_result: str
    priority: str
    type: str
    tags: List[str] = []
    risk: Optional[str] = None
    trace: Optional[str] = None


class UserStory(BaseModel):
    id: str = Field(..., example="US-001")
    feature: str
    title: str
    role: str
    goal: str
    benefit: str
    acceptance_criteria: List[str] = []
    related_test_ids: List[str] = []


class TestPlanResponse(BaseModel):
    document_title: Optional[str] = None
    summary: Optional[str] = None
    user_stories: List[UserStory] = []
    test_cases: List[TestCase]
    assumptions: List[str] = []
    risks: List[str] = []
