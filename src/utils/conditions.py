from ..core.trajectory import Trajectory
from ..core.messages import AIMessage

# ------------------------------------------------------------------
# Boolean value that Agent has finished the task in given trial step.
# Agent is considered to have finished the task if the last message is 'AIMessage' or call 'complete_task' method in ToolCallMessage.
# ------------------------------------------------------------------
def is_agent_finished(trajectory: Trajectory) -> bool:
    if isinstance(trajectory.messages[-1], AIMessage) or "complete_task" in trajectory.messages[-1].content:
        return True
    
    return False