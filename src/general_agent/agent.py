from __future__ import annotations

import json
import logging
import re
import argparse
from datetime import datetime, timezone
from pathlib import Path
from unittest import result
from langgraph.types import Command, interrupt

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages

from operator import add

from general_agent.config import GeneralAgentConfig
from general_agent.prompts import UX_PLANNER_PROMPT, UX_SYSTEM_PROMPT, CONTEXT_ANALYZER_PROMPT, QUESTION_DECIDER_PROMPT
from general_agent.schemas import UXAgentState, ContextAnalysis, QuestionDecision, create_initial_ux_state, UXPlan
from general_agent.tools import GeneralAgentTools

_PROFILES_PATH = Path(__file__).parent / "data" / "profiles.json"


def _load_profiles() -> dict:
    if _PROFILES_PATH.exists():
        with _PROFILES_PATH.open() as f:
            return json.load(f)
    return {"profiles": []}


def _find_profile(user_id: str | None, profile_id: str | None) -> dict:
    if not user_id or not profile_id:
        return {}

    data = _load_profiles()
    profile = next(
        (
            p
            for p in data["profiles"]
            if p.get("owner_user_id") == user_id and p.get("id") == profile_id
        ),
        None,
    )
    return profile or {}


def _build_chat_llm(config: GeneralAgentConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.model_name,
        temperature=config.temperature,
        api_key=config.api_key,
        base_url=config.api_base,
        max_tokens=config.max_tokens,
    )


def _build_context_user_prompt(
    *,
    message: str,
    profile_summary: str,
    session_context: dict,
    task_context: dict,
    compact: bool,
) -> str:
    if compact:
        return f"""
Return only compact JSON that matches the schema exactly.
Keep list fields short and bounded.

Current user request:
{message}

Saved UX/product profile (truncated):
{profile_summary[:1800]}

Existing session context:
{json.dumps(session_context or {}, indent=2)[:1200]}

Existing task context:
{json.dumps(task_context or {}, indent=2)[:1200]}
"""

    return f"""
Current user request:
{message}

Saved UX/product profile:
{profile_summary}

Existing session context:
{json.dumps(session_context or {}, indent=2)}

Existing task context:
{json.dumps(task_context or {}, indent=2)}

Analyze the current request using the saved profile when useful.
Generate session_context and task_context even if some details are missing.
Do not return empty objects unless the request is completely unrelated to UX/product design.
"""


def _invoke_context_analyzer(
    *,
    llm,
    message: str,
    profile_summary: str,
    session_context: dict,
    task_context: dict,
):
    try:
        return llm.invoke(
            [
                ("system", UX_SYSTEM_PROMPT),
                ("system", CONTEXT_ANALYZER_PROMPT),
                (
                    "user",
                    _build_context_user_prompt(
                        message=message,
                        profile_summary=profile_summary,
                        session_context=session_context,
                        task_context=task_context,
                        compact=False,
                    ),
                ),
            ]
        )
    except Exception as exc:
        if exc.__class__.__name__ != "LengthFinishReasonError":
            raise

        logger.warning(
            "Context analysis exceeded model length; retrying with compact prompt."
        )
        return llm.invoke(
            [
                ("system", UX_SYSTEM_PROMPT),
                ("system", CONTEXT_ANALYZER_PROMPT),
                (
                    "user",
                    _build_context_user_prompt(
                        message=message,
                        profile_summary=profile_summary,
                        session_context=session_context,
                        task_context=task_context,
                        compact=True,
                    ),
                ),
            ]
        )


def _profile_summary(profile_data: dict) -> str:
    if not profile_data:
        return "No saved profile data is available."

    return json.dumps(
        {
            "profile_name": profile_data.get("name"),
            "product": profile_data.get("product"),
            "domain": profile_data.get("domain"),
            "target_users": profile_data.get("target_users", []),
            "user_goals": profile_data.get("user_goals", []),
            "pain_points": profile_data.get("pain_points", []),
            "workflows": profile_data.get("workflows", []),
            "constraints": profile_data.get("constraints", []),
            "preferences": profile_data.get("preferences", []),
            "business_context": profile_data.get("business_context", {}),
        },
        indent=2,
    )


def _normalize_context_analysis(result: ContextAnalysis) -> dict:
    return {
        "session_context": result.session_context.model_dump()
        if hasattr(result.session_context, "model_dump")
        else result.session_context,
        "task_context": result.task_context.model_dump()
        if hasattr(result.task_context, "model_dump")
        else result.task_context,
        "missing_context": result.missing_context,
        "confidence": result.confidence,
        # temporary only; later move to question_decider
        "should_ask_questions": bool(result.missing_context) and result.confidence < 0.55,
    }


def run_context_analysis_once(
    *,
    message: str,
    user_id: str | None = None,
    profile_id: str | None = None,
    session_context: dict | None = None,
    task_context: dict | None = None,
    config: GeneralAgentConfig | None = None,
) -> dict:
    """Run context analysis directly without invoking the LangGraph workflow."""
    active_config = config or GeneralAgentConfig.from_env()
    context_config = GeneralAgentConfig(
        model_name=active_config.model_name,
        api_key=active_config.api_key,
        api_base=active_config.api_base,
        temperature=active_config.temperature,
        max_tokens=min(active_config.max_tokens, 1200),
    )
    llm = _build_chat_llm(context_config).with_structured_output(ContextAnalysis)
    profile_data = _find_profile(user_id, profile_id)
    profile_summary = _profile_summary(profile_data)

    result = _invoke_context_analyzer(
        llm=llm,
        message=message,
        profile_summary=profile_summary,
        session_context=session_context or {},
        task_context=task_context or {},
    )

    return _normalize_context_analysis(result)


def profile_loader(state: UXAgentState) -> dict:
    """LangGraph node — load an existing profile."""
    user_id = state.get("user_id")
    profile_id = state.get("profile_id")

    profile = _find_profile(user_id, profile_id)

    if not profile:
        return {"profile_data": {}}
    
    return {"profile_data": profile}




def final_response(state: UXAgentState) -> dict:
    content = json.dumps(
        state,
        indent=2,
        default=str,
    )
    
    return {
        "final_instructions": content,
        "messages": [AIMessage(content=content)],
    }


logger = logging.getLogger(__name__)



class DesignerAgent:
    def __init__(
        self,
        config: GeneralAgentConfig,
        agent_tools: GeneralAgentTools,
    ) -> None:
        self.config = config
        self.system_prompt = UX_SYSTEM_PROMPT
        self.agent_tools = agent_tools
        self._llm = self._build_llm()
        self._tool_llm = self._llm.bind_tools(self.tools) if self.tools else self._llm
        context_config = GeneralAgentConfig(
            model_name=self.config.model_name,
            api_key=self.config.api_key,
            api_base=self.config.api_base,
            temperature=self.config.temperature,
            max_tokens=min(self.config.max_tokens, 1200),
        )
        self._context_analyzer_llm = _build_chat_llm(context_config).with_structured_output(ContextAnalysis)
        self._graph = self.build_graph()

        logger.info(
            "DesignerAgent initialized (model=%s, tools=%s)",
            self.config.model_name,
            list(self.tools_by_name.keys()),
        )

    @property
    def tools(self):
        return list(self.agent_tools.tools.values())

    @property
    def tools_by_name(self):
        return self.agent_tools.tools

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DesignerAgent:
        config = GeneralAgentConfig.from_env()
        agent_tools = GeneralAgentTools(config)
        return cls(config=config, agent_tools=agent_tools)

    def _build_llm(self):
        return _build_chat_llm(self.config)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        message: str,
        thread_id: str,
        user_id: str,
        profile_id: str,
        is_new_session: bool = False,
    ) -> dict:
        """Run the graph. Returns a dict with either questions or the final result.

        Return shapes:
          - {"status": "needs_input", "questions": [...], "missing_context": [...]}
          - {"status": "complete", "result": <final output string>}
        """
        config = {"configurable": {"thread_id": thread_id}}

        if is_new_session:
            graph_input = create_initial_ux_state(
                message=message,
                user_id=user_id,
                profile_id=profile_id,
            )
        else:
            graph_input = {
                "messages": [HumanMessage(content=message)],
                "user_id": user_id,
                "profile_id": profile_id,
            }

        self._graph.invoke(graph_input, config=config)

        # Check if the graph paused at an interrupt
        state = self._graph.get_state(config)
        if state.tasks and any(t.interrupts for t in state.tasks):
            # Extract the interrupt payload (the questions dict)
            interrupt_payload = state.tasks[0].interrupts[0].value
            return {
                "status": "needs_input",
                "questions": interrupt_payload["questions"],
                "missing_context": interrupt_payload["missing_context"],
            }

        # Graph completed — extract final output
        result_state = state.values
        output = result_state.get("final_instructions") or result_state["messages"][-1].content
        return {"status": "complete", "result": output}

    def resume(
        self,
        thread_id: str,
        user_answers: str,
    ) -> dict:
        """Resume the graph after the user answers discovery questions.

        Return shape is the same as run().
        """
        config = {"configurable": {"thread_id": thread_id}}

        self._graph.invoke(Command(resume=user_answers), config=config)

        state = self._graph.get_state(config)
        if state.tasks and any(t.interrupts for t in state.tasks):
            interrupt_payload = state.tasks[0].interrupts[0].value
            return {
                "status": "needs_input",
                "questions": interrupt_payload["questions"],
                "missing_context": interrupt_payload["missing_context"],
            }

        result_state = state.values
        output = result_state.get("final_instructions") or result_state["messages"][-1].content
        return {"status": "complete", "result": output}

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def build_graph(self):
        builder  = StateGraph(UXAgentState)
        
        builder.add_node("profile_loader", profile_loader)
        builder.add_node("context_analyzer", self._context_analyzer)
        builder.add_node("question_decider", self._question_decider)
        builder.add_node("discovery_questions", self._discovery_questions)
        builder.add_node("ux_planner", self._ux_planner)
        builder.add_node("final_response", final_response)

        builder.add_edge(START, "profile_loader")
        builder.add_edge("profile_loader", "context_analyzer")
        builder.add_edge("context_analyzer", "question_decider")

        builder.add_conditional_edges(
            "question_decider",
            self._route_after_question_decider,
            {
                True: "discovery_questions",
                False: "ux_planner",
            },
        )

        builder.add_edge("discovery_questions", "ux_planner")
        builder.add_edge("ux_planner", "final_response")
        builder.add_edge("final_response", END)

        return builder.compile(checkpointer=MemorySaver())

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _has_tool_calls(self, state: UXAgentState) -> bool:
        last = state["messages"][-1]
        return bool(getattr(last, "tool_calls", None))

    def _call_llm(self, state: UXAgentState) -> dict:
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self._llm.invoke(messages)
        return {"messages": [response]}
    
    def _profile_summary(self, profile_data: dict) -> str:
        return _profile_summary(profile_data)
    
    def _context_analyzer(self, state: UXAgentState) -> dict:
        latest_user_message = state["messages"][-1].content
        result = _invoke_context_analyzer(
            llm=self._context_analyzer_llm,
            message=latest_user_message,
            profile_summary=self._profile_summary(state.get("profile_data", {})),
            session_context=state.get("session_context", {}),
            task_context=state.get("task_context", {}),
        )

        return _normalize_context_analysis(result)

    def analyze_context(
        self,
        *,
        message: str,
        user_id: str | None = None,
        profile_id: str | None = None,
        session_context: dict | None = None,
        task_context: dict | None = None,
    ) -> dict:
        """Public helper to run context analysis without the LangGraph flow."""
        return run_context_analysis_once(
            message=message,
            user_id=user_id,
            profile_id=profile_id,
            session_context=session_context,
            task_context=task_context,
            config=self.config,
        )
        
    def _question_decider(self, state: UXAgentState) -> dict:
        """Decides the next question to ask based on the current state."""
        should_ask = bool(state.get("missing_context")) and state.get("confidence", 0.0) < 0.55
        
        return {"should_ask_questions": should_ask}

    def _route_after_question_decider(self, state: UXAgentState) -> bool:
        return bool(state.get("should_ask_questions", False))
    
    def _discovery_questions(self, state: UXAgentState) -> dict:
        """Discovery questions to ask the user when important context is missing."""
        result = self._llm.with_structured_output(QuestionDecision).invoke([
            ("system", UX_SYSTEM_PROMPT),
            ("system", QUESTION_DECIDER_PROMPT),
            ("user", json.dumps({
                "missing_context": state.get("missing_context", []),
                "session_context": state.get("session_context", {}),
                "task_context": state.get("task_context", {}),
            }, indent=2)),
        ])

        # Pause graph execution — the interrupt payload is returned to the caller
        # so they can display the questions. The graph resumes when the caller
        # invokes with Command(resume=<user's answers string>).
        user_answers = interrupt({
            "questions": result.discovery_questions,
            "missing_context": result.missing_context,
        })

        # After resume: user_answers is whatever the caller passed via Command(resume=...)
        return {
            "discovery_questions": result.discovery_questions,
            "missing_context": result.missing_context,
            "messages": [HumanMessage(content=user_answers)],
        }
        
        
        
    def _ux_planner(self, state: UXAgentState) -> dict:
        """Generate a UX plan based on the current context."""
        content = self._llm.with_structured_output(UXPlan).invoke([
            ("system", UX_SYSTEM_PROMPT),
            ("system", UX_PLANNER_PROMPT),
            ("user", json.dumps({
                "profile_data": state.get("profile_data", {}),
                "session_context": state.get("session_context", {}),
                "task_context": state.get("task_context", {}),
                "missing_context": state.get("missing_context", []),
                "confidence": state.get("confidence", 0.0),
                }, indent=2)),
        ])
        logger.info("Generated UX plan: %s", content)
        return {"ux_plan": content.model_dump() if hasattr(content, "model_dump") else content}
         

    def _execute_tool_calls(self, state: UXAgentState) -> dict:
        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        errors: list[str] = []

        for call in last_message.tool_calls:
            tool_name = call["name"]
            args = call.get("args", {})
            selected_tool = self.tools_by_name.get(tool_name)

            if not selected_tool:
                content = f"Unknown tool: {tool_name}"
                errors.append(content)
                tool_messages.append(
                    ToolMessage(content=content, name=tool_name, tool_call_id=call["id"])
                )
                continue

            try:
                result = selected_tool.invoke(args)
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        name=tool_name,
                        tool_call_id=call["id"],
                    )
                )
            except Exception as exc:
                error_msg = f"Tool '{tool_name}' raised an error: {exc}"
                errors.append(error_msg)
                logger.exception("Tool execution failed: %s", tool_name)
                tool_messages.append(
                    ToolMessage(
                        content=error_msg,
                        name=tool_name,
                        tool_call_id=call["id"],
                    )
                )

        return {"messages": tool_messages, "errors": errors}

USER_PROMPT = """I haveto create add a create/edit dashboard page.
I have Charts, Tables, HistoricalChart and HistoricalTables to display using the dashboard
What will be the best way to create the create dashboard functionality?
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Designer agent runner")
    parser.add_argument(
        "--mode",
        choices=["full", "context"],
        default="context",
        help="full = run graph, context = run context analyzer only",
    )
    parser.add_argument("--message", default=USER_PROMPT, help="User request")
    parser.add_argument("--thread-id", default="session-1")
    parser.add_argument("--user-id", default="user_001")
    parser.add_argument("--profile-id", default="profile_001")
    parser.add_argument(
        "--is-new-session",
        action="store_true",
        help="Run graph in new-session mode (full mode only)",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.mode == "context":
        output = run_context_analysis_once(
            message=args.message,
            user_id=args.user_id,
            profile_id=args.profile_id,
        )
        print(json.dumps(output, indent=2, default=str))
    else:
        agent = DesignerAgent.from_env()
        response = agent.run(
            args.message,
            thread_id=args.thread_id,
            user_id=args.user_id,
            profile_id=args.profile_id,
            is_new_session=args.is_new_session,
        )

        if response["status"] == "needs_input":
            print("\n--- Discovery Questions ---")
            for i, q in enumerate(response["questions"], 1):
                print(f"  {i}. {q}")
            print()
            answers = input("Your answers: ")
            response = agent.resume(args.thread_id, answers)

        print(json.dumps(response, indent=2, default=str))


# Activate your venv first
# source .venv/bin/activate

# Default: context analysis mode
# python -m general_agent.agent

# Explicit context mode with a custom message
# python -m general_agent.agent --mode context --message "Your question here"

# Full graph mode (runs the LangGraph workflow)
# python -m general_agent.agent --mode full --user-id user_001 --profile-id profile_001

# Full mode with a new session + custom thread
# python -m general_agent.agent --mode full --is-new-session --thread-id my-thread --user-id user_001 --profile-id profile_001