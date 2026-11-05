import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Search tool
search_tool = TavilySearch(
    max_results=5,
    topic="news",
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

# 2. LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# 3. Prompt
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.

    Summarize the following news into clear bullet points.
    Provide a concise summary.

    News:
    {topic}
    """
)

# 4. Chain
chain = prompt | llm | StrOutputParser()

# 5. Search news
news_result = search_tool.invoke({
    "query": "latest news about AI 2026"
})

# 6. Summarize news
result = chain.invoke({
    "topic": str(news_result)
})

# 7. Print output
print("\n===== AI NEWS SUMMARY =====\n")
print(result)
