class BaseMessage:
    def __init__(
        self, 
        role: str,
        content: str,
    ):
        self.role = role
        self.content = content
    
    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content
        }
    
    def __repr__(self):
        return f"BaseMessage(role={self.role}, content={self.content})"

class UserMessage(BaseMessage):
    def __init__(
        self, 
        content: str
    ):
        super().__init__(
            role="user", content=content)
    
    def __repr__(self):
        return f"UserMessage(content={self.content})"

class AIMessage(BaseMessage):
    def __init__(
        self, 
        content: str
    ):
        super().__init__(role="assistant", content=content)
    
    def __repr__(self):
        return f"AIMessage(content={self.content})"

class ToolCallMessage:
    def __init__(
        self, 
        msg_type: str,
        call_id: str,
        name: str,
        arguments: str
    ):
        self.type = msg_type
        self.call_id = call_id
        self.name = name
        self.arguments = arguments
    
    def to_dict(self):
        return {
            "type": self.type,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments
        }
    
    def __repr__(self):
        return f"ToolCallMessage(type={self.type}, call_id={self.call_id}, name={self.name}, arguments={self.arguments})"

class ToolCallOutputMessage:
    def __init__(
        self, 
        msg_type: str,
        call_id: str,
        output: str
    ):
        self.type = msg_type
        self.call_id = call_id
        self.output = output
    
    def to_dict(self):
        return {
            "type": self.type,
            "call_id": self.call_id,
            "output": self.output
        }
    
    def __repr__(self):
        return f"ToolCallOutputMessage(type={self.type}, call_id={self.call_id}, output={self.output})"