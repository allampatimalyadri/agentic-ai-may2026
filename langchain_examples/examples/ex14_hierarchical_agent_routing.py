#  Hierachical Agents with Decision Routing

# hierarchical_agents.py
# Hierarchical Multi-Agent System — Pure LangChain LCEL (no LangGraph, no tool-wrapping)

from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()


# ============================================================
# Model
# ============================================================
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# ============================================================
# Routing Schema  (structured output — not a tool call)
# ============================================================
class RouteDecision(BaseModel):
    """Supervisor's routing decision."""
    next: Literal["research", "write", "finish"] = Field(
        description="Which agent to invoke next, or 'finish' if done."
    )
    instruction: str = Field(
        description="Specific instruction to pass to the next agent."
    )


# ============================================================
# Sub-Agent Chains  (pure LCEL: prompt | model | parser)
# ============================================================
research_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Provide detailed, factual information on any topic."),
        ("human", "{instruction}"),
    ])
    | model
    | StrOutputParser()
)

writing_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a professional writer. Create clear, concise, and engaging content."),
        ("human", "{instruction}\n\nResearch data to use:\n{research_output}"),
    ])
    | model
    | StrOutputParser()
)


# ============================================================
# Supervisor Chain  (structured output for routing)
# ============================================================
supervisor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", (
            "You are a supervisor coordinating research and writing agents.\n"
            "Routing rules:\n"
            "  - 'research' → if research has NOT been done yet\n"
            "  - 'write'    → if research is done but writing is NOT done\n"
            "  - 'finish'   → if both research and writing are complete\n"
            "Always provide a specific, detailed instruction for the next agent."
        )),
        ("human", "{context}"),
    ])
    | model.with_structured_output(RouteDecision)
)


# ============================================================
# Orchestrator  (while loop handles the cycle — LCEL is DAG-only)
# ============================================================
def orchestrate(inputs: dict) -> str:
    user_task = inputs["task"]
    state = {
        "research_output": None,
        "writing_output": None,
    }

    print("\n========= 0. Orchestration started ============")

    while True:
        # Build context for supervisor
        context = f"""
User task: {user_task}

Research completed: {"Yes" if state["research_output"] else "No"}
Writing completed:  {"Yes" if state["writing_output"] else "No"}
{f"Research gathered:{chr(10)}{state['research_output']}" if state["research_output"] else ""}
""".strip()

        # Supervisor decides next step via structured output (NOT tool calling)
        decision: RouteDecision = supervisor_chain.invoke({"context": context})
        print(f"\n========= Supervisor → '{decision.next}' ============")
        print(f"Instruction: {decision.instruction[:80]}...")

        if decision.next == "research":
            print("\n========= 1. Research Agent working ============")
            state["research_output"] = research_chain.invoke({
                "instruction": decision.instruction,
            })
            print("Research Agent done ✓")

        elif decision.next == "write":
            print("\n========= 2. Writing Agent working ============")
            state["writing_output"] = writing_chain.invoke({
                "instruction": decision.instruction,
                "research_output": state["research_output"] or "",
            })
            print("Writing Agent done ✓")

        elif decision.next == "finish":
            print("\n========= Done ✓ ============")
            return state["writing_output"] or state["research_output"] or "No output generated."


# Wrap as a standard LCEL Runnable → supports .invoke(), .stream(), .batch()
hierarchical_agent = RunnableLambda(orchestrate)


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    result = hierarchical_agent.invoke({
        "task": "Research the benefits of Protein and write a 2-paragraph summary."
    })

    print("\n=== Final Response ===")
    print(result)