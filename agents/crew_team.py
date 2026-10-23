"""
The three agents as a CrewAI crew.

Techniques from the DeepLearning.AI x CrewAI course that are used here:
  * role-based Agents (role, goal, backstory) and Tasks (description, expected output)
  * a sequential Crew with context passing (the Reviewer reads the other two agents' output)
  * Task guardrails: our code verifier validates every output; when it finds a problem,
    CrewAI sends the problem back to the agent and the agent tries again
  * Flows: the surrounding workflow lives in agents/flow.py

Everything numeric stays in code (facts.py); the agents only write language.
"""
import json
import re

from crewai import Agent, Crew, LLM, Process, Task, TaskOutput

from agents import config, grapher, summariser, verifier
from agents.facts import fact_table
from agents.llm import LLMFatalError
from agents.schemas import AgentOutputError, ChartsOut, ReviewOut, SummaryOut, parse

ROLES = {
    "summariser": dict(
        role="Plain-Language CVA Explainer",
        goal="Explain a credit valuation adjustment result to a reader with no finance background, "
             "using only the numbers provided.",
        backstory="You are a risk communicator who turns quantitative results into short, accurate "
                  "plain-English explanations. You never change, round or invent a number."),
    "grapher": dict(
        role="Chart Selection Specialist",
        goal="Choose the charts that best answer the reader's question and caption each in one plain sentence.",
        backstory="You are a data-visualisation specialist who picks from a fixed catalogue of charts "
                  "and never invents numbers."),
    "reviewer": dict(
        role="Fact-Checking Reviewer",
        goal="Find unsupported claims, advice, wrong definitions and unexplained jargon in a draft explanation.",
        backstory="You are a strict editor who protects readers from misleading financial explanations. "
                  "You do not check arithmetic; a separate system does that."),
}


def build_llm() -> LLM:
    """CrewAI LLM pointed at Groq through its OpenAI-compatible endpoint (no LiteLLM needed)."""
    return LLM(model=f"openai/{config.GROQ_MODEL}", base_url=config.GROQ_OPENAI_BASE,
               api_key=config.GROQ_API_KEY, temperature=0)


def extract_json(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise AgentOutputError("no JSON object found in the answer")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise AgentOutputError(f"invalid JSON: {exc}") from exc


def _fatal(exc: BaseException):
    """An HTTP error that retrying cannot fix (bad key, unknown model, bad request)."""
    seen = 0
    while exc is not None and seen < 6:
        if getattr(exc, "status_code", None) in (400, 401, 403, 404):
            return exc
        exc, seen = (exc.__cause__ or exc.__context__), seen + 1
    return None


class CvaCrewTeam:
    """Builds the three agents once and runs one crew per attempt."""

    def __init__(self, llm=None, verbose: bool | None = None):
        self.llm = llm or build_llm()
        self.verbose = config.CREW_VERBOSE if verbose is None else verbose
        self.agents = {
            key: Agent(**spec, llm=self.llm, allow_delegation=False, verbose=self.verbose,
                       max_iter=3, max_retry_limit=1)
            for key, spec in ROLES.items()
        }

    @property
    def description(self) -> dict:
        return {"framework": "CrewAI", "process": "sequential",
                "agents": [a.role for a in self.agents.values()]}

    def run_attempt(self, facts, allowed, question=None, summary=None, charts=None,
                    s_problems=None, c_problems=None) -> dict:
        """One crew run. Only the agents whose output still needs work are included."""
        held = {"summary": None, "charts": None, "review": None, "events": []}

        def guardrail(kind, model, check):
            def fn(output: TaskOutput):
                try:
                    obj = parse(model, extract_json(output.raw))
                except AgentOutputError as exc:
                    held["events"].append({"task": kind, "code": "agent_failed", "message": str(exc)})
                    return False, f"Your answer was not valid ({exc}). Return only the JSON object described."
                issues = check(obj)
                if issues:
                    held["events"] += [{"task": kind, "code": i.code, "message": i.message} for i in issues]
                    return False, "Fix these problems and return the full JSON again: " + " | ".join(i.message for i in issues)
                held[kind] = obj
                return True, output
            return fn

        retries = config.GUARDRAIL_RETRIES
        tasks, run_summary, run_charts = [], summary is None or bool(s_problems), charts is None or bool(c_problems)

        if run_summary:
            user = summariser.build_user(facts, question, summary, s_problems)
            summary_task = Task(
                description=summariser.SYSTEM + "\n\nINPUT (JSON):\n" + json.dumps(user),
                expected_output="One JSON object with headline, explanation, key_numbers and caveats.",
                agent=self.agents["summariser"], guardrail_max_retries=retries,
                guardrail=guardrail("summary", SummaryOut, lambda o: verifier.verify_summary(o, allowed)))
            tasks.append(summary_task)
        if run_charts:
            user = grapher.build_user(facts, question, charts, c_problems)
            charts_task = Task(
                description=grapher.SYSTEM + "\n\nINPUT (JSON):\n" + json.dumps(user),
                expected_output="One JSON object with a charts list.",
                agent=self.agents["grapher"], guardrail_max_retries=retries,
                guardrail=guardrail("charts", ChartsOut, lambda o: verifier.verify_charts(o, allowed)))
            tasks.append(charts_task)

        # The reviewer reads this attempt's new drafts through CrewAI context, and any draft that
        # was already approved earlier is passed inline.
        approved = {}
        if not run_summary:
            approved["summary"] = summary.model_dump()
        if not run_charts:
            approved["charts"] = charts.model_dump()
        review_text = (verifier.REVIEW_SYSTEM + "\n\nFACTS (JSON):\n" + json.dumps(fact_table(facts))
                       + "\n\nThe draft to review is in the context from the previous tasks.")
        if approved:
            review_text += "\nAlready approved parts of the draft (JSON):\n" + json.dumps(approved)
        review_task = Task(
            description=review_text,
            expected_output='One JSON object: {"pass": boolean, "issues": [{"target": "summary" or "charts", "message": string}]}',
            agent=self.agents["reviewer"], context=list(tasks), guardrail_max_retries=1,
            guardrail=guardrail("review", ReviewOut, lambda o: []))
        tasks.append(review_task)

        crew = Crew(agents=list(self.agents.values()), tasks=tasks, process=Process.sequential,
                    verbose=self.verbose, tracing=False)
        error = None
        try:
            crew.kickoff()
        except Exception as exc:  # classified below
            fatal = _fatal(exc)
            if fatal is not None:
                raise LLMFatalError(f"LLM error {getattr(fatal, 'status_code', '')}: {str(fatal)[:300]}") from exc
            error = exc

        usage = getattr(crew, "usage_metrics", None)
        return {
            "summary": held["summary"] or summary, "charts": held["charts"] or charts,
            "review": held["review"], "events": held["events"],
            "error": None if error is None else str(error)[:300],
            "guardrail_exhausted": error is not None and "guardrail" in str(error).lower(),
            "calls": getattr(usage, "successful_requests", 0) or 0,
            "tokens": getattr(usage, "total_tokens", 0) or 0,
        }
