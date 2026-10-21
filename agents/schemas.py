"""Validated shapes of what each agent must return."""
from pydantic import BaseModel, Field, ValidationError


class AgentOutputError(Exception):
    """An agent returned JSON that does not match its schema."""


class KeyNumber(BaseModel):
    fact_key: str
    label: str = Field(min_length=1, max_length=60)


class SummaryOut(BaseModel):
    headline: str = Field(min_length=1, max_length=200)
    explanation: str = Field(min_length=40, max_length=2500)
    key_numbers: list[KeyNumber] = Field(min_length=3, max_length=5)
    caveats: list[str] = Field(min_length=1, max_length=4)


class ChartPick(BaseModel):
    id: str
    caption: str = Field(min_length=1, max_length=300)


class ChartsOut(BaseModel):
    charts: list[ChartPick] = Field(min_length=1, max_length=3)


class ReviewIssue(BaseModel):
    target: str  # "summary" or "charts"
    message: str


class ReviewOut(BaseModel):
    pass_: bool = Field(alias="pass")
    issues: list[ReviewIssue] = []


def parse(model, data):
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise AgentOutputError(
            f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}") from exc
