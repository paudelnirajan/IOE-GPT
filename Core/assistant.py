from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel
from .state import State
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable
        logger.debug(f"Initialized Assistant with runnable: {type(runnable).__name__}")

    def __call__(self, state: State, config: RunnableConfig):
        logger.debug("Assistant called with new state")
        while True:
            result = self.runnable.invoke(state)
            logger.debug(f"Runnable result: has_tool_calls={bool(result.tool_calls)}, has_content={bool(result.content)}")

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                logger.info("Empty result detected, requesting real output")
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }

    def __init__(self, **data):
        super().__init__(**data)
        logger.info(f"CompleteOrEscalate initialized: cancel={self.cancel}, reason={self.reason}")