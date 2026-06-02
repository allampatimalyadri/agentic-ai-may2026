from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.graph import MermaidDrawMethod

# 1. Define shared state
class AgentState(TypedDict):
    query: str
    needs_clarification: bool
    response: str


# 2. Nodes
def research_node(state: AgentState) -> AgentState:
    """
    Simulates understanding the user query.
    Decides if the query is clear enough.
    """
    query = state["query"]
    print(f"🔍 Analyzing query: {query}")

    # Simple heuristic: vague questions trigger clarification
    vague_keywords = ["what", "tell me", "explain"]
    needs_clarification = any(word in query.lower() for word in vague_keywords) and len(query.split()) < 6

    return {
        **state,
        "needs_clarification": needs_clarification,
        "response": f"I found some info about '{query}'"
    }


def answer_node(state: AgentState) -> AgentState:
    """
    Generates final answer when query is clear.
    """
    print("✅ Generating final answer")

    return {
        **state,
        "response": f"""{state['response']}. 
        Here is a clear and complete answer.
        .......
        ......
        .....
        .....
        """
    }


def clarify_node(state: AgentState) -> AgentState:
    """
    Asks user for more details when query is vague.
    """
    print("❓ Asking for clarification")

    return {
        **state,
        "response": f"Your question '{state['query']}' is a bit unclear. Can you provide more details?"
    }


# 3. Router (decision maker)

def route(state: AgentState) -> str:
    """
    Decides next step based on clarity of query.
    """
    return "clarify" if state["needs_clarification"] else "answer"


# 4. Build graph
workflow = StateGraph(AgentState)

workflow.add_node("research", research_node)
workflow.add_node("answer", answer_node)
workflow.add_node("clarify", clarify_node)

workflow.add_edge(START, "research")

workflow.add_conditional_edges(
    "research",
    route,
    {
        "answer": "answer",
        "clarify": "clarify"
    }
)

workflow.add_edge("answer", END)
workflow.add_edge("clarify", END)


# 5. Compile
app = workflow.compile()


# Generate and save the graph visualization
graph_image = app.get_graph().draw_mermaid_png()
with open("examples/basic_workflow.png", "wb") as f:
    f.write(graph_image)

# 6. Run examples

print("\n--- Example 1: Clear Query ---")
result1 = app.invoke({
    "query": "How does LangGraph work step by step?",
    "needs_clarification": False,
    "response": ""
})
print("Final:", result1["response"])


# print("\n--- Example 2: Vague Query ---")
# result2 = app.invoke({
#     "query": "Explain LangGraph",
#     "needs_clarification": False,
#     "response": ""
# })
# print("Final:", result2["response"])