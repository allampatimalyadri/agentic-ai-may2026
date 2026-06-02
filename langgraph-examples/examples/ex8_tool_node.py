# REFER: tool calling explainer.md file in the same directory
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests

load_dotenv()

# =========================
# 1. LLM
# =========================
llm = ChatOpenAI(model="gpt-5.4")


# =========================
# 2. Tools
# =========================

@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a math expression"""
    print("calling calculator_tool")
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

@tool
def weather_tool(location: str) -> dict:
    """Get current weather for a city and return temperature"""
    print("calling weather_tool")
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_resp = requests.get(geo_url, params={"name": location, "count": 1}, timeout=10)
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return {"error": f"Could not find location: {location}"}

        place = geo_data["results"][0]
        lat, lon = place["latitude"], place["longitude"]
        city_name = place.get("name", location)

        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_resp = requests.get(
            weather_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
            },
            timeout=10,
        )

        data = weather_resp.json()
        temp = data["current_weather"]["temperature"]

        return {
            "city": city_name,
            "temperature": temp
        }

    except Exception as e:
        return {"error": str(e)}


tools = [calculator_tool, weather_tool]

# Bind tools
llm_with_tools = llm.bind_tools(tools)


# =========================
# 3. State
# =========================

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


# =========================
# 4. Agent Node (LLM only)
# =========================

def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    
    return {
        "messages": [response]
    }

# =========================
# 5. Conditional Routing
# =========================
def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"
    
    return END


# =========================
# 6. Tool Node (prebuilt)
# =========================

# Initializing ToolNode with tools and configuration.
tool_node = ToolNode(tools)


# =========================
# 7. Build Graph
# =========================
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    ["tools", END]
)

workflow.add_edge("tools", "agent")

app = workflow.compile()
# Generate and save the graph visualization
graph_image = app.get_graph().draw_mermaid_png()
with open("examples/ex8_tool_node.png", "wb") as f:
    f.write(graph_image)


# =========================
# 8. Run Example
# =========================
result = app.invoke({
    "messages": [
        {"role": "user", "content": """What is the weather in Chennai 
         and multiply that by 2?"""}
    ]
})

print("\nFinal Answer:\n")
print(result["messages"][-1].content)