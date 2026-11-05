
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

result = get_weather.invoke({"city": "Lucknow"})
print(result)


# Initialize Tavily Search
tavily_client = TavilySearch(
    max_results=5,
    topic="news",
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str):
    """Search for the latest news related to a given city."""
    try:
        # Search latest news about the given city
        response = tavily_client.invoke({
            "query": f"latest news about {city}"
        })

        # Return search results
        return response

    except Exception as e:
        return f"Error fetching news: {str(e)}"


# Test the tool

result = get_news.invoke({"city": "Lucknow"})
print(result)


# Initialize the LLM with tools
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=SecretStr(groq_api_key) if groq_api_key else None,
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=500
)

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tools = llm.bind_tools([get_weather, get_news])


# Agent LOOP
messages = []

print("city intelligence system")
print("Type 'exit' to quit the program.")

while True:
    user_input = input("Enter a city name: ").strip()

    if user_input.lower() == "exit":
        break

    if not user_input:
        continue

    # Start a fresh conversation for each city query
    # This prevents old incomplete tool-call messages from carrying over
    messages = [
        SystemMessage(content="""
You are a city intelligence assistant.

Use get_weather when the user asks about current weather,
temperature, humidity, or wind.

Use get_news when the user asks about latest or recent news
related to a city.

Use both tools if the user asks for both weather and news.

Do not use tools for greetings or general knowledge questions.
If no tool is needed, answer normally.
"""),
        HumanMessage(content=user_input)
    ]

    try:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            # Check whether both APIs are requested
            tool_names = [tool_call["name"] for tool_call in response.tool_calls]

            if "get_weather" in tool_names and "get_news" in tool_names:
                confirm = input(
                    "Weather aur news dono APIs call karni hain. Continue? (yes/no): "
                ).strip().lower()

                if confirm not in ["yes", "y"]:
                    print("Agent: Okay, dono API calls cancel kar di.")
                    continue

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]
                tool_call_id = tool_call["id"]

                if tool_name in tools:
                    tool_result = tools[tool_name].invoke(tool_input)
                    print(f"Tool '{tool_name}' output: {tool_result}")
                else:
                    tool_result = f"Tool '{tool_name}' not found."

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id
                    )
                )

            final_response = llm_with_tools.invoke(messages)
            messages.append(final_response)
            print(f"Agent: {final_response.content}")

        else:
            print(f"Agent: {response.content}")

    except Exception as e:
        print(f"Error: {e}")


        # manually call the tools if the agent didn't do it automatically