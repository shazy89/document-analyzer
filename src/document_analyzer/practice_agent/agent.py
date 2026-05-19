
import operator
from typing import TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from typing_extensions import Annotated

from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import tool

from document_analyzer.analyzer_agent.config import DocumentAnalyzerConfig


@tool
def capitalize_first_letter(s: str) -> str:
    """Capitalize the first letter of a string."""
    if not s:
        return s
    return s[0].upper() + s[1:]

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

tools = [add, multiply, divide]

tools_by_name = {tool.name: tool for tool in tools}

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

class PracticeAgent:
    def __init__(self, config: DocumentAnalyzerConfig,):
        self.config = config
        
        self._llm = self._llm_model()
        
    def _llm_model(self):
        return ChatOpenAI(model=self.config.model_name, temperature=0, api_key=self.config.api_key, base_url=self.config.api_base).bind_tools(tools_by_name)    
    
    def llm_call(self, state: list[AnyMessage]) -> AnyMessage:
        messages = state.get("messages", [])
        llm_calls = state.get("llm_calls", 0)
        
        system_prompt = SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
        
        messages = [system_prompt] + messages
        content = self._llm.invoke(messages)
        
        return {"messages": [content], "llm_calls": llm_calls + 1}
        
