from dotenv import load_dotenv
load_dotenv()
import os

from langchain.tools import tool
from langchain_groq import ChatGroq
from pydantic import SecretStr

from rich import print

#1 creating a tool using the @tool decorator

@tool
def get_text_lenght(text: str) -> int:
    """Calculate the length of the given text."""
    return len(text)


llm = ChatGroq(
    model="openai/gpt-oss-20b", 
    temperature=0,
    api_key=SecretStr(os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None
)

#2 tool binding with LLM

llm_with_tool = llm.bind_tools([get_text_lenght]) #binding the tool to the LLM

result = llm_with_tool.invoke("Hello, my dog is cute")
print(result.content) #print the content of the result  


#not completed yet properly
