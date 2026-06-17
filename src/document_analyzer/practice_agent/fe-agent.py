from typing import Literal, TypedDict

from pathlib import Path
from langchain_community.chat_models import ChatOpenAI
from typing_extensions import Annotated
from langgraph.graph.message import BaseMessage, add_messages
from langchain_community.tools import tool
from operator import add
import json
from document_analyzer.analyzer_agent.config import DocumentAnalyzerConfig
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

SYSTEM_PROMPT = """
You are Wealth Wing, a practical financial assistant that helps users understand where their money is going.

Use the available tools when the user asks about specific transactions, merchants, categories, totals, averages, or spending patterns.

Be clear, direct, and do not invent transaction data.
"""

SUMMARIZER_PROMPT = """
You are Wealth Wing, a practical financial assistant that helps users understand where their money is going.

You will receive a list of financial transactions. Each transaction may include:
- date: The transaction date, for example "2024-01-15"
- description: The merchant or transaction description
- amount: The transaction amount
- category: The spending category, for example "Food", "Utilities", "Entertainment"
- type: Optional. Either "income" or "expense"

Your job is to analyze the transactions and explain the user's spending clearly.

Focus on:
1. Where the most money went
2. Which categories had the highest spending
3. Any unusual or repeated transactions
4. Spending patterns, such as frequent small purchases, large one-time expenses, recurring bills, subscriptions, or lifestyle spending
5. Practical insights the user can act on

Rules:
- Be clear, direct, and useful.
- Do not shame the user.
- Do not give generic advice like "spend less" unless the data supports it.
- Do not invent facts that are not visible in the transactions.
- If the data is limited, say that clearly.
- Treat refunds, income, transfers, and expenses differently when possible.
- If categories look wrong or too broad, mention that better categorization would improve the analysis.
- Use simple language, not accounting jargon.

Your response should follow this structure:

## Spending Summary
Give a short overview of total spending and the main spending areas.

## Top Categories
List the categories where the user spent the most money and explain what that suggests.

## Notable Patterns
Identify repeated purchases, subscriptions, large expenses, frequent merchants, or unusual activity.

## Practical Takeaways
Give 2-4 specific recommendations based on the transactions.

## Final Read
End with a short, honest interpretation of the user's financial behavior for this period.

If there is not enough transaction data to identify a trend, avoid over-analyzing and say that more data is needed.
"""

_PROFILES_PATH = Path(__file__).parent / "data.json"


def _load_transactions() -> dict:
    if _PROFILES_PATH.exists():
        with _PROFILES_PATH.open() as f:
            return json.load(f)
    return {"transactions": []}

@tool
def _find_transactions(description: str) -> dict:
    """Find transactions matching a description.
       Example user prompt: "What transactions do I have related to Starbucks?" -> description="Starbucks"
    Args:
        description (str): The description to search for.

    Returns:
        dict: A dictionary containing the matching transactions.
    """
    transactions = _load_transactions().get("transactions", [])
    transactions_data = [txn for txn in transactions if description.lower() in txn["description"].lower()]
    total_spent = sum(txn["amount"] for txn in transactions_data)
    average_transaction = total_spent / len(transactions_data) if transactions_data else 0
    return {"transactions": transactions_data,
            "total_spent": total_spent,
            "average_transaction": average_transaction}
    
class Transaction(TypedDict):
    date: str
    description: str
    amount: float
    category: str
    type: Literal["income", "expense"]


class TransactionSearchResult(TypedDict):
    transactions: list[Transaction]
    total_spent: float
    average_transaction: float
   

class FEAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    transaction_search_result: TransactionSearchResult
    summary: str
    errors: Annotated[list[str], add]

@tool
def transactions_summarizer(state: FEAgentState, _llm: ChatOpenAI) -> str:
    """Summarize a list of transactions into a human-readable format."""
    response = _llm.invoke(
        [   SystemMessage(content=SYSTEM_PROMPT),
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(state["transaction_search_result"]["transactions"])),
        ]
    )

    return {
        "summary": response.content
    }
    
@tool
def final_answer(state: FEAgentState) -> str:
    """Generate a final answer for the user based on the transaction search result and summary."""
    summary = state.get("summary", "")
    transactions = state["transaction_search_result"]["transactions"]

    if not transactions:
        return "I couldn't find any transactions matching that description."

    return (f"Based on the transactions I found, here's what I can tell you:\n\n{summary}")   
    
    

tools = [transactions_summarizer, _find_transactions]
tools_by_name = {tool.name: tool for tool in tools}







class TransactionSearchResult(TypedDict):
    transactions: list[dict]
    total_spent: float
    average_transaction: float


class FEAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    transaction_search_result: TransactionSearchResult
    summary: str
    errors: Annotated[list[str], add]


class FeAgent:
    def __init__(self, config: DocumentAnalyzerConfig, checkpointer: MemorySaver) -> None:
        self.config = config
        self.checkpointer = checkpointer

        self._llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.config.api_key,
            base_url=self.config.api_base,
        )

        self._tool_llm = self._llm.bind_tools(tools)
        self.graph = self._build_graph()

    def llm_call(self, state: FEAgentState) -> dict:
        try:
            response = self._tool_llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    *state.get("messages", []),
                ]
            )

            return {
                "messages": [response]
            }

        except Exception as e:
            return {
                "errors": [str(e)]
            }

    def _build_graph(self):
        builder = StateGraph(FEAgentState)

        builder.add_node("llm_call", self.llm_call)
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "llm_call")

        builder.add_conditional_edges(
            "llm_call",
            tools_condition,
            {
                "tools": "tools",
                END: END,
            },
        )

        builder.add_edge("tools", "llm_call")

        return builder.compile(checkpointer=self.checkpointer)

    def invoke(self, user_message: str, thread_id: str) -> FEAgentState:
        return self.graph.invoke(
            {
                "messages": [
                    HumanMessage(content=user_message)
                ]
            },
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            },
        ) 
            
            
            
            
            
            
            