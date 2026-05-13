from langchain_core.tools import tool

from .config import DocumentAnalyzerConfig
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
class AgentState(TypedDict):
    messages: Annotated[MessagesState, "List of messages in the conversation history"]


@tool
def calculate_similarity(text1: str, text2: str) -> float:
    """Placeholder for a similarity calculation between two pieces of text."""
    
    return f"Calculated similarity between '{text1[:30]}...' and '{text2[:30]}...': 0.75"


def keyword_extraction(text: str) -> list[str]:
    """Placeholder for keyword extraction from text."""
    return ["keyword1", "keyword2", "keyword3"]

def web_search(query: str) -> list[dict]:
    """Placeholder for a web search tool."""
    search_tool = DuckDuckGoSearchRun(max_results=2, output_format="list")
    return search_tool.run(query)


class DocumentAnalyzerAgent:
    def __init__(self, config: DocumentAnalyzerConfig):
        self.config = config
        self.tools = {
            "calculate_similarity": calculate_similarity,
            "keyword_extraction": keyword_extraction,
            "web_search": web_search,
        }
        
    
    def _build_llm(self) -> ChatOpenAI:
        model = ChatOpenAI(model=self.config.model_name, temperature=self.config.temperature, api_key=self.config.api_key, api_base=self.config.api_base)
        
        return model.bind_tools(self.tools)
    
    def build_graph(self) -> StateGraph[AgentState]:
        graph = StateGraph[AgentState]()
        
        graph.add_state(START, {"messages": []})
        graph.add_state(END, {"messages": []})
        # Additional states and transitions would be defined here based on the agent's workflow
        return graph    
    