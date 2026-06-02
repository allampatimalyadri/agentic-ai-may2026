from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

my_agent = create_agent(
  model="openai:gpt-4o-mini", #brain
  system_prompt="You are a helpful assistant.", # roles and goals
)

response = my_agent.invoke(
  {
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }
)
print(response["messages"][-1].text)