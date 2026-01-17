from abc import ABC, abstractmethod
from typing import Any, Sequence
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.messages import AIMessage, ToolMessage

from langgraph.graph.state import CompiledStateGraph

from appworld import AppWorld

from ..state import State

class BaseAgent(ABC):
    def __init__(
        self,
        env: AppWorld,
        system_prompt: str,
        model_config:dict[str, Any] = {
            'model' : 'gpt-4o',
            'temperature' : 0.0,
            'stream_usage' : True
        }
    ):
        self.env = env
        self.system_prompt = system_prompt
        self.model_config = model_config

        openai_client = ChatOpenAI(**model_config)
        self.openai_client_with_tools = openai_client.bind_tools(self._get_tools())

        self.agent: CompiledStateGraph = self._build_agent()

    def _get_tools(self):

        # ----------------------------------------------------------------------------
        # Define Tools
        # ----------------------------------------------------------------------------
        # define execute action tool
        class ActionToolArgsSchema(BaseModel):
            """
            Arugment Schema for code execution tool. This tool execute code and return result message.
            """

            code: str = Field(
                ...,
                description="python code to perform certain task or retrieve information form environment.",
                json_schema_extra={
                    'examples' : [
                        {
                            'example' : 'print(apis.api_docs.show_app_descriptions())',
                            'description' : 'To get a list of apps that are available to agent.'
                        },
                        {
                            'example' : "print(apis.api_docs.show_api_descriptions(app_name='spotify'))",
                            'description' : 'To get the list of apis under any app listed above, e.g. spotify'
                        },
                        {
                            'example' : "print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))",
                            'description' : "To get the specification of a particular api, e.g. spotify app's login api"
                        }
                    ]
                }
            )
        @tool(args_schema=ActionToolArgsSchema)
        def action_tool(
            code:str
        ) -> str:
            """
            Excecute code that Agent generate to perform certain taskor retrieve information form environment. 
            
            This tool execute code and return result message.
            """
            
            try:
                tool_result = f"{self.env.execute(code)}"
                print(f"[Tool | Action] ✅ Action is performed successfully.")
            except Exception as error:
                print(f"[Tool | Action] ⛔️ Action Failed.")
                raise error

            return tool_result
        
        # return tool list
        return [action_tool]
    


    def _get_tool_node(self):
        # get `action_tool`
        tool_list: Sequence[tool] = self._get_tools()
        for _tool in tool_list:
            if _tool.name == 'action_tool':
                action_tool = _tool

        # ----------------------------------------------------------------------------
        # Define Tools
        # ----------------------------------------------------------------------------
        def _tools(state: State):
            last_msg: AIMessage = state['messages'][-1]
            tool_messages = []

            for tool_call in last_msg.tool_calls:
                if tool_call['name'] == 'action_tool':
                    try:
                        tool_message: ToolMessage = ToolMessage(
                            content=action_tool.invoke(tool_call['args']),
                            tool_call_id=tool_call['id']
                        )
                        tool_messages.append(tool_message)
                    except Exception as error:
                        raise error
            
            return {
                'messages' : tool_messages
            }
        
        return _tools
    

    def _get_should_continue(self):

        # Conditional Edge (Should Continue)
        # ==============================================================================================================
        def _should_continue(state: State):
            for msg in state['messages']:
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if 'complete_task' in  tool_call['args']['code']:
                            print("[Conditional Edge] 🎯 Agent Complete Task (Actor used `complete_task`)")
                            return 'end'
            
            return 'tools'
        # ==============================================================================================================
        
        return _should_continue


    @abstractmethod
    def _get_actor_node(self):
        # ----------------------------------------------------------------------------
        # Define Actor Node
        # ----------------------------------------------------------------------------
        raise NotImplementedError()
    

    @abstractmethod
    def _build_agent(self):
        raise NotImplementedError()
    

    def invoke(self, state: State):
        return self.agent.invoke(state)