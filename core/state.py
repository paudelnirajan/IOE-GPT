from langgraph.graph import MessagesState
from typing import Any, TypedDict

class State(MessagesState):
    query: str
    context: dict[str, Any] = {}  # Add context field with default empty dict