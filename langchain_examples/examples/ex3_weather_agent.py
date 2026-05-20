from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests

load_dotenv()

# 1. create a tool 
# tool is a function that the agent can call. 
# tool must must have a name and description. 
# tool must return the data
# tool can have parameters. 
# tool in langchain must have a @tool decorator.
@tool
def get_weather(latitude: str, longitude: str) -> dict:
    """get the weather for a given city."""
    print("Given latitude is: ", latitude)
    print("Given longitude is: ", longitude)
    # let's make api call to open-meteo weather api and get the weather for the given city.
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true")
    current_weather = response.json()
    return current_weather

# Let's create weather agent
weather_agent = create_agent(
  model="google_genai:gemini-3.1-pro-preview", #brain
  tools=[get_weather], # register the tool with the agent
  system_prompt="""You are a weather assistant.
  You must provide accurate weather information for the given city. 
  If the city is invalid or fictional, inform the user accordingly.""", # roles and goals
)

response = weather_agent.invoke(
  {
    "messages": [
      {"role": "user", "content": "What is the weather in New York?"}
    ]
  }
)

print(response["messages"][-1].text)