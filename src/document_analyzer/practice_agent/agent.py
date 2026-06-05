
import operator
import os
from typing import TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from document_analyzer.analyzer_agent.agent import AgentState
from document_analyzer.analyzer_agent.config import DocumentAnalyzerConfig
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

from dotenv import load_dotenv
load_dotenv()


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
    def __init__(self, config: DocumentAnalyzerConfig, checkpointer: MemorySaver) -> None:
        self.system = """
You are a helpful assistant that performs arithmetic operations.

Rules:
- If a calculation is needed, call the correct tool.
- After receiving the tool result, answer in plain English.
- Do not prefix your answer with words like 'assistant' or 'final'.
"""
        self.config = config
        self.checkpointer = checkpointer
        
        self._llm = self._llm_model()
        
        
    def _llm_model(self):
        return ChatOpenAI(model=self.config.model_name, temperature=0, api_key=self.config.api_key, base_url=self.config.api_base).bind_tools(tools)    
    
    def llm_call(self, state: list[AnyMessage]) -> AnyMessage:
        messages = state.get("messages", [])
        llm_calls = state.get("llm_calls", 0)
        
        system_prompt = SystemMessage(
                        content=self.system
                    )
        
        messages = [system_prompt] + messages
        content = self._llm.invoke(messages)
        
        return {"messages": [content], "llm_calls": llm_calls + 1}
    
    def take_aktion(self, state: list[AnyMessage]) -> list[AnyMessage]:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        llm_calls = state.get("llm_calls", 0) + 1

    
        result = []
        if not tool_calls:
            return {"messages": [], "llm_calls": llm_calls}
        
        for t in tool_calls:
            tool = tools_by_name[t["name"]]
            observation = tool.invoke(t.get("args", {}))
            result.append(ToolMessage(content=str(observation), tool_call_id=t["id"]))
        
        return {"messages": result, "llm_calls": llm_calls}
    
    def should_continue(self, state: MessagesState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", []):
            return "action"
        return END

    def build_graph(self):
        graph = StateGraph(MessagesState)
        
        graph.add_node("llm", self.llm_call)
        graph.add_node("action", self.take_aktion)
        
        graph.add_conditional_edges("llm", self.should_continue)
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        
        return graph.compile(checkpointer=self.checkpointer)
        
        
config = DocumentAnalyzerConfig.from_env()        
agent = PracticeAgent(config=config, checkpointer=memory)
graph = agent.build_graph()

thread4 = {"configurable": {"thread_id": "1234"}}
messages = [HumanMessage(content="What is 5 multiplied by 3?")]

for event in graph.stream({"messages": messages, "llm_calls": 0}, thread4):
    print(f"Event: {event}")

    if "llm" in event:
        ai_message = event["llm"]["messages"][-1]

        # Only log final answer when there are no tool calls
        if not getattr(ai_message, "tool_calls", []):
            print("Final answer:", ai_message.content)