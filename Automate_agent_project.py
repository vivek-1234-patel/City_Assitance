import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import requests
from pydantic import SecretStr
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

# ---------- Page config ----------
st.set_page_config(
    page_title="City Assistant",
    page_icon="🌆",
    layout="wide",
)

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        color: #94a3b8;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Clients ----------
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ---------- Tools ----------
@tool
def get_weather(city: str) -> str:
    """Get current weather information for a city using Open-Meteo."""
    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=15,
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Could not find location: {city}"

        location = geo_data["results"][0]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=15,
        )
        weather_response.raise_for_status()
        current = weather_response.json()["current"]

        return (
            f"Current weather in {location['name']}:\n"
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


@tool
def get_news(city: str) -> str:
    """Get latest news about a city."""
    try:
        response = tavily_client.search(
            query=f"latest news in {city}",
            topic="news",
            search_depth="basic",
            max_results=3,
        )
    except Exception as e:
        return f"News search failed: {e}"

    results = response.get("results", [])
    if not results:
        return f"No news found for {city}"

    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        news_list.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...")

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


TOOLS = [get_weather, get_news]

# ---------- LLM + Agent ----------
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=SecretStr(groq_api_key) if groq_api_key else None,
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=500,
)

agent = create_agent(
    llm,
    tools=TOOLS,
    system_prompt="You are a helpful city assistant.",
)


# ---------- Helpers ----------
def run_agent(query: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    content = result["messages"][-1].content
    # Some models return content as a list of blocks; join the text parts
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.caption("Available tools:")
    st.markdown("- 🌦️ Weather\n- 📰 News")
    st.divider()
    require_approval = st.toggle("Require approval before running", value=True)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()


# ---------- Header ----------
st.markdown('<div class="main-title">City Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Weather and city news, all in one place</div>',
    unsafe_allow_html=True,
)


# ---------- Chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- Approval flow ----------
if st.session_state.pending:
    query = st.session_state.pending
    st.warning(f"The agent wants to run tools for: **{query}**")
    col1, col2 = st.columns(2)
    if col1.button("✅ Approve", use_container_width=True):
        with st.spinner("Thinking..."):
            try:
                answer = run_agent(query)
            except Exception as e:
                answer = f"Something went wrong: {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.pending = None
        st.rerun()
    if col2.button("❌ Deny", use_container_width=True):
        st.session_state.messages.append(
            {"role": "assistant", "content": "Tool call denied by user."}
        )
        st.session_state.pending = None
        st.rerun()


# ---------- Chat input ----------
user_input = st.chat_input("Ask about a city... (e.g. Delhi weather)")

if user_input and not st.session_state.pending:
    st.session_state.messages.append({"role": "user", "content": user_input})
    if require_approval:
        st.session_state.pending = user_input
        st.rerun()
    else:
        with st.spinner("Thinking..."):
            try:
                answer = run_agent(user_input)
            except Exception as e:
                answer = f"Something went wrong: {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()