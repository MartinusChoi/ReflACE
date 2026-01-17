from typing import Annotated, Sequence
from operator import add

from langchain.agents.middleware import AgentState


# -----------------------------------------------------------------------------------------------------
# Agent State
# -----------------------------------------------------------------------------------------------------
class State(AgentState):
    # fields for reflexion agent
    evaluation: str
    reflections: Annotated[Sequence[str], add]

    # field for track token usages
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    total_tokens: Annotated[int, add]