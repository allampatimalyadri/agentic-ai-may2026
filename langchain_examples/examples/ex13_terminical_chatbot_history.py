# this example uses runnablemessagehistory to keep track of the conversation history and display it in the terminal.

import os
import uuid
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini", #brain
)

# Session ID
session_id = str(uuid.uuid4())

# Store chat histories
store = {}

# Function to get/create history for a session
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap LLM with message history
chatbot = RunnableWithMessageHistory(
    llm,
    get_session_history
)

print("🤖 Chatbot started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    response = chatbot.invoke(
        [HumanMessage(content=user_input)],
        config={
            "configurable": {
                "session_id": session_id
            },
            "run_name": "terminal_chat",
            "tags": ["chatbot", "terminal", "v1.2"],
            "metadata": {
                "user_id": "user_001",
                "session_id": session_id,
                "interface": "cli"
            }
        }
    )

    print(f"Bot: {response.content}\n")

# Key improvements over ex12_terminal_chatbot.py:

# Uses RunnableWithMessageHistory
# Removes manual messages.append(...)
# Automatically maintains conversation history per session_id
# Uses InMemoryChatMessageHistory for session-based memory management