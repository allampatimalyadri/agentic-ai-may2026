# Modern LangGraph Conversational Chatbot
#
# Features:
# - Proper conversational history
# - Short-term memory only
# - LangGraph-native messages
# - ChatGPT-style terminal loop
# - Structured message architecture
#
# NOTE:
# This example demonstrates SHORT-TERM MEMORY.
# Conversation history exists only during runtime.
# Nothing is persisted across sessions.

from typing import TypedDict, Annotated, List

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.graph.message import add_messages

load_dotenv()


# ==========================================
# 1. LLM
# ==========================================

llm = ChatOpenAI(
    model="gpt-5.4",
    temperature=0.7
)


# ==========================================
# 2. State
# ==========================================

class ChatState(TypedDict):
    # Proper conversational history
    messages: Annotated[List, add_messages]


# ==========================================
# 3. Chat Node
# ==========================================

def chat_node(state: ChatState):

    print("\n🤖 Thinking...\n")

    system_prompt = """
    You are a helpful AI assistant.

    Maintain conversational context naturally.
    """

    # Build messages
    messages = [
        SystemMessage(content=system_prompt)
    ] + state["messages"]

    # Invoke LLM
    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# ==========================================
# 4. Build Graph
# ==========================================

workflow = StateGraph(ChatState)

workflow.add_node("chat",chat_node)

workflow.add_edge(START, "chat")

workflow.add_edge("chat", END)

app = workflow.compile()


# ==========================================
# 5. Terminal Chat Loop
# ==========================================

print("\n========================================")
print("🤖 LangGraph ChatGPT Terminal")
print("Type 'exit' or 'quit' to stop")
print("========================================\n")


# Short-term memory state
state = {
    "messages": []
}


while True:

    user_input = input("🧑 You: ").strip()

    if not user_input:
        continue

    # Exit condition
    if user_input.lower() in ["exit", "quit"]:

        print("\n👋 Goodbye!\n")

        break

    # Add user message
    state["messages"].append(
        HumanMessage(content=user_input)
    )

    # Invoke graph
    state = app.invoke(state)

    # Latest AI response
    last_ai_message = state["messages"][-1]

    print(f"\n🤖 AI: {last_ai_message.content}\n")