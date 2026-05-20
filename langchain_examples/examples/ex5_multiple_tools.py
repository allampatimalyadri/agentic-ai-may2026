# Task 1: call get_weather tool 
# Task 2: call web_search tool 
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
from tavily import TavilyClient # uv add tavily-python
import requests
import os

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_weather_tool(latitude: str, longitude: str) -> dict:
    """get the weather for a given city."""
    print("Given latitude is: ", latitude)
    print("Given longitude is: ", longitude)
    # let's make api call to open-meteo weather api and get the weather for the given city.
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true")
    current_weather = response.json()
    return current_weather

@tool
def web_search_tool(query: str) -> dict:
    """ Get real time updates with web search powered by Tavily."""
    print("Given query is: ", query)
    search_results = tavily_client.search(
        query = query, 
        search_depth = "advanced",
        search_results = 5
    )
    return search_results


# Let's create weather agent
general_purpose_agent = create_agent(
  model="google_genai:gemini-3.1-pro-preview", #brain
  tools=[get_weather_tool, web_search_tool], # register the tool with the agent
  system_prompt="""You are a helpful assistant capable of giving weather updates and 
  real-time information from the web.
  You must provide accurate weather information for the given city's latitude and longitude.
  If the city is invalid or fictional, inform the user accordingly.
  You are given access to the right tools to get the weather information and real-time updates 
  from the web.
  """, # roles and goals
)

response = general_purpose_agent.invoke(
  {
    "messages": [
      # {"role": "user", "content": "What is the weather in New York?"}
      # {"role": "user", "content": "What is the latest ai news?"}
      {"role": "user", "content": "What is the weather in New York and give me latest news about it"}
    ]
  }
)

print(response["messages"][-1].text)