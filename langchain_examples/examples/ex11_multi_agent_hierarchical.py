# hierarchical multi-agent system example

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool     
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm_api_key = os.getenv("GOOGLE_API_KEY")

if not llm_api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    google_api_key=llm_api_key,
) # Initialize the LLM 

# ---- Define specialized sub-agents ----
# Research Agent: searches or retrieves information
research_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""You are a research assistant. Provide detailed,
    factual information on any topic.""",
)

# Let's expose this research agent over a tool
@tool
def call_research_agent_tool(query: str) -> str:
    """Call the research agent to gather information on a topic/query."""
    print("========= 1. Planning to invoke research agent ============")
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

# Writing Agent: writes or summarizes content
writing_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a professional writer. " \
    "Create clear, concise, and engaging content.",
)

# Lets expose this writing agent over a tool
@tool
def call_writing_agent_tool(query: str) -> str:
    """Call the writing agent to write or summarize content."""
    print("========= 2. Planning to invoke writing agent ============")
    result = writing_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

# ---- Define supervisor agent ----
supervisor_agent = create_agent(
    model=model,
    tools=[call_research_agent_tool, call_writing_agent_tool],
    system_prompt=(
        """You are a supervisor agent. You coordinate tasks between specialized agents:
        First ask the research agent to gather necessary information.
        - Use call_research_agent_tool for research and information gathering.
        - Use call_writing_agent_tool for writing, editing, or summarizing.
        Delegate tasks to the appropriate agent and combine their outputs to
        answer the user."""
    ),
)

# ---- Run the multi-agent system ----
if __name__ == "__main__":
    user_task = "Research the benefits of Protein and write a 2-paragraph summary."
    
    print("========= 0. Planning to invoke supervisor agent ============")
    result = supervisor_agent.invoke({
        "messages": [{"role": "user", "content": user_task}]
    })
    
    print("=== Supervisor's Final Response ===")
    print(result["messages"][-1].content)