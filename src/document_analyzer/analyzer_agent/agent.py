from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from operator import add
from document_analyzer.analyzer_agent.config import DocumentAnalyzerConfig
from document_analyzer.analyzer_agent.prompts import SYSTEM_PROMPT
from document_analyzer.analyzer_agent.tools import AnalyzerAgentTools
from document_analyzer.services.together_client import TogetherChatService
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    search_queries: Annotated[list[str], add]
    found_links: Annotated[list[str], add]
    scraped_urls: Annotated[list[str], add]
    skipped_urls: Annotated[list[str], add]
    normalized_jobs: Annotated[list[dict], add]

    errors: Annotated[list[str], add]
    step: str


class DocumentAnalyzerAgent:
    def __init__(
        self,
        config: DocumentAnalyzerConfig,
        agent_tools: AnalyzerAgentTools,
        service: TogetherChatService,
    ) -> None:
        self.config = config
        self.service = service
        self.system_prompt = SYSTEM_PROMPT

        self.agent_tools = agent_tools
        self.tools = list(agent_tools.tools.values())
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        self._llm = self._build_llm()

        logger.info(
            "DocumentAnalyzerAgent initialized (model=%s, tools=%d)",
            self.config.model_name,
            len(self.tools),
        )

    def _build_llm(self):
        model = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.config.api_key,
            base_url=self.config.api_base,
        )

        return model.bind_tools(self.tools)

    def build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("llm", self._call_llm)
        graph.add_node("action", self._execute_tool_calls)

        graph.set_entry_point("llm")

        graph.add_conditional_edges(
            "llm",
            self._has_tool_calls,
            {
                True: "action",
                False: END,
            },
        )

        graph.add_edge("action", "llm")

        return graph.compile(checkpointer=MemorySaver())

    def _has_tool_calls(self, state: AgentState) -> bool:
        last_message = state["messages"][-1]
        return bool(getattr(last_message, "tool_calls", None))

    def _call_llm(self, state: AgentState) -> AgentState:
        messages = state["messages"]

        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages

        response = self._llm.invoke(messages)

        return {"messages": [response]}

    def _execute_tool_calls(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1]    

        tool_messages = []
        found_links = []
        normalized_jobs = []
        errors = [] 

        for call in last_message.tool_calls:
            tool_name = call["name"]
            args = call.get("args", {}) 

            selected_tool = self.tools_by_name.get(tool_name)   

            if not selected_tool:
                content = f"Unknown tool: {tool_name}"
                errors.append(content)  

                tool_messages.append(
                    ToolMessage(
                        content=content,
                        name=tool_name,
                        tool_call_id=call["id"],
                    )
                )
                continue    

            try:
                result = selected_tool.invoke(args) 

                if isinstance(result, list):
                    for item in result:
                        if not isinstance(item, dict):
                            continue    

                        if item.get("source_link"):
                            found_links.append(item["source_link"]) 

                        if item.get("normalized"):
                            normalized_jobs.append(item)    

                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        name=tool_name,
                        tool_call_id=call["id"],
                    )
                )   

            except Exception as e:
                content = f"Error executing tool '{tool_name}': {e}"
                errors.append(content)  

                tool_messages.append(
                    ToolMessage(
                        content=content,
                        name=tool_name,
                        tool_call_id=call["id"],
                    )
                )   

        return {
            "messages": tool_messages,
            "found_links": found_links,
            "normalized_jobs": normalized_jobs,
            "errors": errors,
            "step": "tools_executed",
        }
    
    
# Factory function to create an instance of DocumentAnalyzerAgent with the necessary configuration and tools    
def create_job_hunter_agent(config: DocumentAnalyzerConfig) -> DocumentAnalyzerAgent:
    service = TogetherChatService(api_key=config.api_key, default_model=config.model_name)

    agent_tools = AnalyzerAgentTools(config=config, service=service)
    agent_tools.register_tool("search_jobs", agent_tools.search_content)

    return DocumentAnalyzerAgent(
        config=config,
        agent_tools=agent_tools,
        service=service,
    )    
    