from typing import Annotated
from operator import add

from langchain.agents.middleware import AgentState

from appworld import AppWorld


# -----------------------------------------------------------------------------------------------------
# ReAct Agent
# -----------------------------------------------------------------------------------------------------
class ReActState(AgentState):

    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    total_tokens: Annotated[int, add]