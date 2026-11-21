import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_SECONDS = 180


@tool
def hello_world() -> str:
    """Return a greeting from a local tool."""
    return "Hello from your local agent tool!"


async def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add your Google Gemini API key to the .env file."
        )

    model = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    agent = create_agent(model=model, tools=[hello_world])
    prompt = input("Ask the agent something (try: Use your hello_world tool): ")
    print("Thinking... (waiting for Gemini)", flush=True)
    print("\nAgent: ", end="", flush=True)
    async for message_chunk, metadata in agent.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "model" and message_chunk.text:
            print(message_chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
