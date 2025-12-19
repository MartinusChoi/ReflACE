from typing import List, Union
from .messages import BaseMessage, ToolCallMessage, ToolCallOutputMessage

class Trajectory:
    def __init__(
        self,
        messages: List[BaseMessage | ToolCallMessage | ToolCallOutputMessage]
    ):
        self.messages = messages
    
    def append(
        self,
        message: Union[BaseMessage, ToolCallMessage, ToolCallOutputMessage]
    ):
        self.messages.append(message)
    
    def to_context(self):
        return [msg.to_dict() for msg in self.messages]
    
    def __repr__(self):
        history = "\n\n".join([msg.__repr__() for msg in self.messages])
        return f"Trajectory(messages={history})"