import os
import uuid
from dotenv import load_dotenv
from langchain.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

session_id = str(uuid.uuid4())

print("🤖 Chatbot started. Type 'exit' to quit.\n")

messages = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    messages.append(HumanMessage(content=user_input))

    response = llm.invoke(
        messages,
        config={
            "run_name": "terminal_chat",
            "tags": ["chatbot", "terminal", "v1.2"],
            "metadata": {
                "user_id": "user_001",
                "session_id": session_id,
                "interface": "cli"
            }
        }
    )

    messages.append(AIMessage(content=response.content))

    print(f"Bot: {response.content}\n")