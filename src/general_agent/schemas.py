from typing import Annotated, TypedDict, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing import List
from operator import add


class QuestionDecision(BaseModel):
    should_ask_questions: bool
    missing_context: List[str] = Field(default_factory=list)
    discovery_questions: List[str] = Field(default_factory=list)


from typing import Annotated, Any, Dict, List, Optional, TypedDict
from operator import add

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class UXAgentState(TypedDict):
    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Identity / profile selection
    user_id: str
    profile_id: Optional[str]
    profile_name: Optional[str]

    # Loaded durable profile context
    profile_data: Dict[str, Any]

    # Temporary working context
    session_context: Dict[str, Any] 
    task_context: Dict[str, Any] # This is the current request after analysis.

    # Discovery / control flow
    missing_context: List[str]
    should_ask_questions: bool
    discovery_questions: List[str]

    # Agent outputs
    ux_plan: Dict[str, Any]
    ui_plan: Dict[str, Any]

    # Final Copilot-ready output
    final_instructions: Optional[str]

    # Diagnostics
    errors: Annotated[list[str], add]
    


def create_initial_ux_state(
    *,
    message: str,
    user_id: str,
    profile_id: str,
) -> UXAgentState:
    if not message.strip():
        raise ValueError("message is required")

    if not user_id:
        raise ValueError("user_id is required")

    if not profile_id:
        raise ValueError("profile_id is required")

    return {
        "messages": [HumanMessage(content=message)],

        "user_id": user_id,
        "profile_id": profile_id,
        "profile_name": "",

        "profile_data": {},

        "session_context": {},
        "task_context": {},

        "missing_context": [],
        "should_ask_questions": False,
        "discovery_questions": [],

        "ux_plan": {},
        "ui_plan": {},

        "final_instructions": "",

        "errors": [],
    }    