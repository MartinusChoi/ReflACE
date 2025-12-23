from ..core.trajectory import Trajectory
from ..core.messages import AIMessage

# ------------------------------------------------------------------
# Boolean value that Agent has finished the task in given trial step.
# Agent is considered to have finished the task if the last message is 'AIMessage' or call 'complete_task' method in ToolCallMessage.
# ------------------------------------------------------------------
def is_agent_finished(trajectory: Trajectory) -> bool:
    last_msg = trajectory.messages[-1]
    
    if isinstance(last_msg, AIMessage):
        return True

    if hasattr(last_msg, 'content') and last_msg.content and "complete_task" in last_msg.content:
        return True
    
    if hasattr(last_msg, 'name') and last_msg.name == "complete_task":
        return True
    
    return False