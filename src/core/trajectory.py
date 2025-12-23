from typing import List, Union
from .messages import BaseMessage, UserMessage, AIMessage, ToolCallMessage, ToolCallOutputMessage

class Trajectory:
    def __init__(
        self,
        messages: List[BaseMessage | UserMessage | AIMessage | ToolCallMessage | ToolCallOutputMessage]
    ):
        self.messages = messages
    
    def append(
        self,
        message: Union[BaseMessage, UserMessage, AIMessage, ToolCallMessage, ToolCallOutputMessage]
    ):
        self.messages.append(message)
    
    def reset(self):
        self.messages = []
    
    def to_chat_prompt(self):
        return [msg.to_dict() for msg in self.messages]
    
    def to_str(self) -> str:
        trajectory_str = "<current trajectory>\n"
        for prompt in self.to_chat_prompt():
            trajectory_str += f"{str(prompt)}\n"
        trajectory_str += "</current trajectory>\n\n"
        return trajectory_str
    
    def __repr__(self):
        history = "\n\n".join([msg.__repr__() for msg in self.messages])
        return f"Trajectory(messages={history})"