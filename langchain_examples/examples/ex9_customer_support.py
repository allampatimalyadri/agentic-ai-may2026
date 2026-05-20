# This example demonstrates how to create a customer support agent for an e-commerce company 
# using LangChain. 
# The agent can understand customer questions, recommend products, 
# and check stock availability using a defined tool.
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 

load_dotenv()


# -----------------------------
# 1. Create reusable prompt template
# -----------------------------
sales_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert sales assistant for an e-commerce company.

        Your responsibilities:
        - Help users understand products
        - Recommend suitable products
        - Write persuasive and clear responses
        - Be concise and professional

        Store Name: {store_name}
        Brand Tone: {brand_tone}
        """
    ),
    (
        "user",
        """
        Customer Question:
        {question}
        """
    )
])

# -----------------------------
# 2. Define a tool
# -----------------------------
@tool
def check_stock(product_name: str) -> str:
    """Check stock availability for a product."""
    
    inventory = [
        {
            "id": 1,
            "name": "LG 4K TV",
            "status": "in_stock",
            "quantityLeft": 2
        },
        {
            "id": 2,
            "name": "Sony Headphones",
            "status": "low_stock",
            "quantityLeft": 2
        },
        {
            "id": 3,
            "name": "Apple iPhone 17",
            "status": "out_of_stock"
        }
    ]

    for item in inventory:
        if item["name"] == product_name:   
            return item
    return "Product not found"


# -----------------------------
# 3. Initialize model
# -----------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# 4. Generate system prompt dynamically
# -----------------------------
final_system_prompt = sales_prompt.invoke({
    "store_name": "ABC Electronics",
    "brand_tone": "Premium and Friendly",
    "question": "placeholder"
}).messages[0].content


# -----------------------------
# 5. Create agent
# -----------------------------
customer_support_agent = create_agent(
    model=model,
    tools=[check_stock],
    system_prompt=final_system_prompt
)

# -----------------------------
# 6. Invoke agent
# -----------------------------
response = customer_support_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Give me the details of LG 4K TV and check if it's available in stock."
        }
    ]
})

print(response["messages"][-1].text)