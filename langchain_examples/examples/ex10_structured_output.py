# structured output 
import os
from langchain.agents import create_agent
from langchain_core.tools import tool         
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

# ── 1. Define structured output schema ──────────────────────────────────────
class NewsArticle(BaseModel):
    """A single news article."""
    title: str = Field(description="Title of the article")
    source: str = Field(description="Publisher or source name")
    summary: str = Field(description="Brief summary of the article")
    url: str = Field(description="URL of the article, empty string if unavailable")

class AINewsResponse(BaseModel):
    """Structured response for AI news queries."""
    topic: str = Field(description="The specific topic of AI news covered")
    articles: List[NewsArticle] = Field(description="List of relevant articles found")
    overall_summary: str = Field(description="High-level summary of current AI news")


# ── 2. Decorate tools with @tool ─────────────────────────────────────────────
@tool
def get_weather_tool(city: str) -> str:
    """A simple tool that fetches weather information for a given city."""
    print(f"Fetching weather information for {city}...")
    api_key = os.getenv("OPENWEATHER_API_KEY")
    api_url = os.getenv("OPENWEATHER_API_URL")
    # TODO: Make an API call using api_key and api_url
    return f"In {city} temperature is 25°C."


@tool
def websearch_tool(query: str) -> str:
    """Use this for general web search queries."""
    print(f"Performing web search for {query}...")
    response = client.search(
        query=query,
        search_depth="advanced"
    )
    return response


# ── 3. Pass response_format to create_agent ──────────────────────────────────
general_purpose_agent = create_agent(
    model="openai:gpt-5.4",
    tools=[get_weather_tool, websearch_tool],
    response_format=AINewsResponse,             # ← structured output schema
    system_prompt="""You are a helpful assistant.
    You are given two tools - get_weather_tool which fetches weather information for a given city,
    and websearch_tool which performs general web searches.
    If the tool outputs are not sufficient to answer the query,
    say you are unable to answer instead of making up a response.
    """
)


response = general_purpose_agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Tell me about latest AI news"}
        ]
    }
)



news: AINewsResponse = response["structured_response"]
# print(news) # pydantic format of the structured response
print(news.model_dump_json(indent=2)) # for exact json output 

# you can also access the raw json output from the model if you want to see the exact format 
# before parsing it into the pydantic model
news_in_json = response["messages"][-1].text
print(news_in_json) # raw json output from the model

# print(f"Topic: {news.topic}")
# print(f"\nOverall Summary:\n{news.overall_summary}")
# print(f"\nArticles ({len(news.articles)}):")
# for article in news.articles:
#     print(f"\n  [{article.source}] {article.title}")
#     print(f"  {article.summary}")
#     if article.url:
#         print(f"  🔗 {article.url}")

