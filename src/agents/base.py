from abc import ABC, abstractmethod
from typing import Any, Union
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.tools import tool

from langgraph.graph.state import CompiledStateGraph

from appworld import AppWorld

from ..state import ReActState, ReflexionState

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
        
        # get tool list cache
        self.tool_list = self._get_tool_list()

        # create llm client
        openai_client = ChatOpenAI(**model_config)
        # bind tools to llm client
        self.openai_client_with_tools = openai_client.bind_tools(self.tool_list)

        # build agent instance
        self.agent: CompiledStateGraph = self._build_agent()

    def _get_tool_list(self):

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
        
        return [action_tool]

    @abstractmethod
    def _build_agent(self):
        raise NotImplementedError()
    

    def invoke(self, state: Union[ReActState, ReflexionState]):
        return self.agent.invoke(state)