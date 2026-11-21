from langchain.agents import create_agent as build_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph

from document_pipeline.config import (
    GENERATION_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    get_gemini_api_key,
)
from document_pipeline.retriever import DocumentRetriever


class DocumentAgent:
    """Create and run an agent that uses a document retriever tool."""

    def __init__(self, retriever: DocumentRetriever) -> None:
        self.retriever = retriever
        self.agent: CompiledStateGraph | None = None

    def _create_retrieval_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.retriever.retrieve_documents,
            name="retrieve_documents",
            description=(
                "Search the indexed documents for information relevant to a query. "
                "Returns matching text chunks with source metadata."
            ),
        )

    def create_agent(self) -> CompiledStateGraph:
        """Create a LangChain agent with document retrieval as a tool."""
        if self.retriever.vector_store is None:
            raise RuntimeError("No Chroma vector store is available to search.")

        model = ChatGoogleGenerativeAI(
            model=GENERATION_MODEL,
            google_api_key=get_gemini_api_key(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        self.agent = build_agent(
            model=model,
            tools=[self._create_retrieval_tool()],
            system_prompt=(
                "You answer questions about the indexed documents. Call the "
                "retrieve_documents tool to find relevant context before answering. "
                "Base your answer only on the returned document content, and say "
                "when the documents do not contain the answer."
            ),
        )
        return self.agent

    def run(self, prompt: str) -> None:
        """Send a prompt to the agent and print its final response."""
        if not prompt.strip():
            raise ValueError("Agent prompt must not be empty.")
        if self.retriever.vector_store is None:
            print("\nNo Chroma vector store is available to answer the prompt.")
            return

        print("\nAgent response:")
        print(self.answer(prompt))

    def answer(self, prompt: str) -> str:
        """Return the agent's answer without printing it."""
        if not prompt.strip():
            raise ValueError("Agent prompt must not be empty.")
        if self.retriever.vector_store is None:
            raise RuntimeError("No Chroma vector store is available to answer.")

        agent = self.agent or self.create_agent()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        messages = result.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            raise RuntimeError("The retrieval agent returned no final AI response.")

        return messages[-1].text
