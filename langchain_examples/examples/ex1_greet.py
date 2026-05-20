from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os 

load_dotenv()

print("Gemini API Key:", os.getenv("GEMINI_API_KEY"))

model = init_chat_model('google_genai:gemini-3.1-pro-preview')
response = model.invoke("Hello, how are you?")
print(response)

# to run 
# python examples/ex1_greet.py