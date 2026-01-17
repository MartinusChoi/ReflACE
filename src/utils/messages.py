from langchain.messages import AIMessage, ToolMessage, HumanMessage, AnyMessage
from typing import Sequence

def pretty_print_messages(messages: Sequence[AnyMessage]) -> None:
    for msg in messages:
        print("======="*20)

        if isinstance(msg, HumanMessage):
            print("👤 Human Instruction")
            print("======="*20)
            print(f"{msg.content}")

        elif isinstance(msg, AIMessage):
            print("💼 Agent's Tool Call")
            for tool_call in msg.tool_calls:
                print("-------"*20)
                print(f"Code : \n{tool_call['args']['code']}")
                print("-------"*20)

        elif isinstance(msg, ToolMessage):
            print("🌏 Observation")
            print("======="*20)
            print(f"\n{msg.content}\n")

        print("======="*20)
        print("\n\n")