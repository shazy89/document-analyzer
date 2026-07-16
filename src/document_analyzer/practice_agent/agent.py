#!/usr/bin/env python3
from typing import Literal, TypedDict
from datetime import datetime 
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from document_analyzer.analyzer_agent.config import DocumentAnalyzerConfig
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

from dotenv import load_dotenv
load_dotenv()

spending_category = [{
    "date": "2026-05-25",
    "description": "PS&G",
    "amount": 120.42,
    "category": "Utilities",
    "type": "expense"
  },
    {
    "date": "2026-06-25",
      "description": "PS&G",
    "amount": 320.42,
    "category": "Utilities",
    "type": "expense"
  },
    {
    "date": "2026-07-25",
    "description": "PS&G",
    "amount": 300.42,
    "category": "Utilities",
    "type": "expense"
  },
    {
    "date": "2026-08-25",
    "description": "PS&G",
    "amount": 140.42,
    "category": "Utilities",
    "type": "expense"
  },
    {
    "date": "2026-09-25",
    "description": "PS&G",
    "amount": 120.42,
    "category": "Utilities",
    "type": "expense"
  }
]
class DataRequest(BaseModel):
    intent: Literal[
        "spending_by_category",
        "total_spending"
    ]
    period: datetime | None = None
    category: str | None = None

class UIBlock(BaseModel):
    component: Literal[
        "summary_cards",
        "bar_chart",
        "transactions_table",
    ]
    title: str
    data_request: DataRequest

class UIPlan(BaseModel):
    message: str
    blocks: list[UIBlock] = Field(min_length=1, max_length=3)

@tool
def spending_by_category():
    """
    Tool to get spending by category.
    """
    return    spending_category

@tool
def total_spending():
    """
    Tool to get total spending.
    """
    total = sum(item["amount"] for item in spending_category)
    return total

 


class UiAgent:
    def __init__(self, config: DocumentAnalyzerConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.api_base
        )

    def builder_graph(self):
        builder = StateGraph(UIPlan)

        builder.add_node("tools", ToolNode([spending_by_category, total_spending]))    

        return builder.compile()
    

    