"""
LLM Study Assistant - TAE-2: Application Development in LLM
Built with the OpenAI Agents SDK (the tool studied in TAE-1).

Features (maps to the Agents SDK primitives from the TAE-1 report):
  - Agent      : triage, benchmark, coding and research agents
  - Handoff    : triage agent delegates to the right specialist
  - Guardrail  : input guardrail blocks off-topic messages
  - Session    : SQLite session keeps conversation history
  - Tools      : custom function tool + built-in WebSearchTool
"""

import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    SQLiteSession,
    TResponseInputItem,
    WebSearchTool,
    function_tool,
    input_guardrail,
)

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --------------------------------------------------------------------------
# Custom function tool: explains the 5 benchmarks covered in the TAE-1 report
# --------------------------------------------------------------------------
BENCHMARKS = {
    "mmlu": "MMLU (Massive Multitask Language Understanding): multiple-choice test "
    "over 57+ academic and professional subjects; measures broad knowledge.",
    "gpqa": "GPQA (Graduate-Level Google-Proof Q&A): 448 PhD-level science questions "
    "in biology, chemistry and physics; measures deep scientific reasoning.",
    "humaneval": "HumanEval: 164 hand-written Python problems checked by unit tests "
    "(pass@1); measures short-function code generation.",
    "swe-bench": "SWE-bench Verified: real resolved GitHub issues; the model must edit "
    "a repository so hidden tests pass; measures agentic software engineering.",
    "truthfulqa": "TruthfulQA: ~800 questions where humans often hold false beliefs; "
    "measures truthfulness and resistance to common misconceptions.",
}


@function_tool
def explain_benchmark(name: str) -> str:
    """Explain one LLM benchmark.

    Args:
        name: One of mmlu, gpqa, humaneval, swe-bench, truthfulqa.
    """
    key = name.strip().lower().replace(" ", "-").replace("_", "-")
    if key in ("swe-bench-verified", "swebench"):
        key = "swe-bench"
    return BENCHMARKS.get(key, f"Unknown benchmark '{name}'. Choose from: {', '.join(BENCHMARKS)}.")


# --------------------------------------------------------------------------
# Input guardrail: keep the assistant on-topic (AI / LLM / programming / study)
# --------------------------------------------------------------------------
class TopicCheck(BaseModel):
    is_off_topic: bool
    reason: str


guardrail_agent = Agent(
    name="Topic guardrail",
    model=MODEL,
    instructions=(
        "Decide whether the user's message is unrelated to AI, large language models, "
        "programming, computer science or studying. Greetings and short follow-ups "
        "count as on-topic. Set is_off_topic=true only for clearly unrelated requests."
    ),
    output_type=TopicCheck,
)


@input_guardrail
async def topic_guardrail(
    ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_off_topic,
    )


# --------------------------------------------------------------------------
# Specialist agents
# --------------------------------------------------------------------------
benchmark_agent = Agent(
    name="Benchmark Tutor",
    model=MODEL,
    handoff_description="Explains LLM benchmarks like MMLU, GPQA, HumanEval, SWE-bench, TruthfulQA.",
    instructions=(
        "You teach LLM evaluation benchmarks. Use the explain_benchmark tool for the "
        "five standard benchmarks, then add a short, clear example of why it matters."
    ),
    tools=[explain_benchmark],
)

coding_agent = Agent(
    name="Coding Helper",
    model=MODEL,
    handoff_description="Writes and explains code, debugs programs.",
    instructions=(
        "You are a patient programming tutor. Give correct, minimal code with a short "
        "explanation. Use fenced code blocks."
    ),
)

research_agent = Agent(
    name="Research Agent",
    model=MODEL,
    handoff_description="Answers questions needing up-to-date information from the web.",
    instructions=(
        "Use web search for current information, then summarise in a few concise "
        "points and mention the sources you used."
    ),
    tools=[WebSearchTool()],
)

triage_agent = Agent(
    name="Triage Agent",
    model=MODEL,
    instructions=(
        "You are the front desk of an LLM study assistant. Hand off to the Benchmark "
        "Tutor for benchmark questions, the Coding Helper for programming questions, "
        "and the Research Agent for current-events or latest-information questions. "
        "Answer simple greetings yourself."
    ),
    handoffs=[benchmark_agent, coding_agent, research_agent],
    input_guardrails=[topic_guardrail],
)


# --------------------------------------------------------------------------
# CLI chat loop
# --------------------------------------------------------------------------
async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")

    session = SQLiteSession("study-assistant")  # remembers the conversation
    print("LLM Study Assistant (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        try:
            result = await Runner.run(triage_agent, user_input, session=session)
            print(f"\n[{result.last_agent.name}] {result.final_output}\n")
        except InputGuardrailTripwireTriggered:
            print("\nSorry, I can only help with AI, LLMs, programming and study topics.\n")


if __name__ == "__main__":
    asyncio.run(main())
