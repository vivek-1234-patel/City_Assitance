
from urllib import response

from dotenv import load_dotenv
load_dotenv()
import os
import requests
import json
 
from langchain_groq import ChatGroq
from langchain.tools import tool
from pydantic import SecretStr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_tavily import TavilySearch
from rich import print
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

tavily_client = TavilySearch(
    max_results=5,
    topic="news",
    tavily_api_key=os.getenv("TAVILY_API_KEY") 
)

#1. Create a tool to get weather information
@tool
def get_weather(city: str):
    """Get current weather information for a city using Open-Meteo."""
    try:
        # Step 1: Convert city name to coordinates
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=15
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Could not find location: {city}"

        location = geo_data["results"][0]
        latitude = location["latitude"]
        longitude = location["longitude"]
        city_name = location["name"]

        # Step 2: Get current weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            "timezone": "auto"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=15
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data["current"]

        return (
            f"Current weather in {city_name}:\n"
            f"Temperature: {current['temperature_2m']}°C\n"
            f"Feels like: {current['apparent_temperature']}°C\n"
            f"Humidity: {current['relative_humidity_2m']}%\n"
            f"Wind speed: {current['wind_speed_10m']} km/h\n"
            f"Weather code: {current['weather_code']}"
        )

    except requests.exceptions.RequestException as e:
        return f"Weather request failed: {e}"
    except (KeyError, IndexError, TypeError) as e:
        return f"Could not process weather data: {e}"



# Initialize Tavily Search
tavily_client = TavilySearch(
    max_results=5,
    topic="news",
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    
    response = tavily_client.invoke({"query": f"latest news in {city}"})

 # Return search results
    results = response.get("results", [])
    
    if not results:
        return f"No news found for {city}"
    
    news_list = []
    
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

# Initialize the LLM with tools
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=SecretStr(groq_api_key) if groq_api_key else None,
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=500
)


@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)  

agent = create_agent(
    llm,
    tools = [get_weather,get_news],
    system_prompt= "you are a helpful city assistant.",
    middleware= [human_approval]
)

print("City Agent | type exit to quit")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break 
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    print("bot : ", result['messages'][-1].content )