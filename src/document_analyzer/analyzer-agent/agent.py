import logging

from langchain_core.tools import tool

from .config import DocumentAnalyzerConfig
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[MessagesState, "List of messages in the conversation history"]



def web_search(query: str) -> list[dict]:
    """Search the web and return structured search results."""
    logger.debug("web_search called query=%s", query)

    search_tool = DuckDuckGoSearchResults(
        max_results=5,
        output_format="list",
    )

    results = search_tool.invoke(query)

    logger.debug("web_search completed")
    return results

def scrape_web_page(url: str) -> str:
    """Load readable text content from a web page."""
    logger.debug("scrape_web_page called url=%s", url)

    loader = WebBaseLoader(url)
    docs = loader.load()

    logger.debug("scrape_web_page completed")
    return "\n\n".join(doc.page_content for doc in docs)


class DocumentAnalyzerAgent:
    def __init__(self, config: DocumentAnalyzerConfig):
        self.config = config
        self.tools = {
            "web_search": web_search,
            "scrape_web_page": scrape_web_page,
        }
        self.system_prompt = SYSTEM_PROMPT
        self._llm = self._build_llm()
        logger.info(
            "DocumentAnalyzerAgent initialized (model=%s, tools=%d)",
            self.config.model_name,
            len(self.tools),
        )
        
    
    def _build_llm(self) -> ChatOpenAI:
        logger.debug(
            "Building LLM client (model=%s, temperature=%s)",
            self.config.model_name,
            self.config.temperature,
        )
        model = ChatOpenAI(model=self.config.model_name,
                           temperature=self.config.temperature,
                           api_key=self.config.api_key,
                           api_base=self.config.api_base)
        
        return model.bind_tools(self.tools)
    
    def build_graph(self) -> StateGraph[AgentState]:
        logger.debug("Building state graph")
        graph = StateGraph[AgentState]()
        
        
        graph.add_state(END, {"messages": []})
        # Additional states and transitions would be defined here based on the agent's workflow
        logger.debug("State graph initialized with END state")
        return graph
    
    def _has_tool_calls(self, messages: AgentState) -> bool:
        """Check if any message in the conversation history contains a tool call."""
        for msg in messages["messages"]:
            if "tool_calls" in msg and msg["tool_calls"]:
                logger.debug("Tool call found in message history")
                return True
        logger.debug("No tool calls found in message history")
        return False
    
    def _call_llm(self, state: AgentState) -> AgentState:
        """Call the LLM with the current state and return the updated state."""
        logger.debug("Calling LLM (messages=%d)", len(state["messages"]))

        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages

        try:
            response = self._llm.invoke(messages=messages)
        except Exception:
            logger.exception("LLM invocation failed")
            raise

        logger.debug(
            "LLM call completed (has_tool_calls=%s)",
            bool(getattr(response, "tool_calls", None)),
        )
        # Process the response and update the state accordingly
        # This is a placeholder for the actual logic to handle the LLM's response
        return {"messages": state["messages"] + [response]}
    
    def _execute_tool_calls(self, state: AgentState) -> AgentState:
        results: list[ToolMessage] = []
        """Execute any tool calls found in the messages and update the state with results."""
        logger.debug("Checking for tool calls to execute")
        
        for call in state["messages"][-1].tool_calls:
            tool_name = call["name"]
            logger.info("Executing tool | name=%s args=%s", tool_name, call["args"])
            
            if not tool_name in self.tools:
                logger.warning("Tool not found: %s", tool_name)
                content = (
                    f"Unknown tool: {tool_name}. Available tools: {', '.join(self.tools.keys())}"
                )
            else:
                args = self._normalise_args(call["args"])
                try:
                    content = str(self._registry.get(tool_name).invoke(args))
                except Exception as e:
                    logger.exception("Error executing tool '%s'", tool_name)
                    content = f"Error executing tool '{tool_name}': {e}"
            results.append(ToolMessage(content=content, tool_name=tool_name))
            
        return {"messages": results}    

    
    @staticmethod
    def _normalise_args(args) -> dict:
        """Coerce tool arguments into the expected {'query': str} shape."""
        if isinstance(args, str):
            return {"query": args}
        if isinstance(args, dict) and "query" not in args:
            str_vals = [v for v in args.values() if isinstance(v, str)]
            return {"query": str_vals[0]} if str_vals else {"query": str(args)}
        return args
    