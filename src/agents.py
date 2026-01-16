from typing import Sequence, Annotated, Any
from pydantic import Field, BaseModel

from langchain_openai import ChatOpenAI
from langchain.messages import AnyMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import tool

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from appworld import AppWorld

from .prompt.react.system_prompt import REACT_SYSTEM_PROMPT
from .state import ReActState



# --------------------------------------------------------------------------------------------------------
# ReAct Agent
# --------------------------------------------------------------------------------------------------------
def create_react_agent(
    model_config:dict[str, Any] = {
        'model' : 'gpt-4o',
        'temperature' : 0.0,
        'stream_usage' : True
    },
    env: AppWorld = None
) -> CompiledStateGraph:
    
    if not env:
        raise ValueError('AppWorld Environment instance is not passed.')

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
            tool_result = f"{env.execute(code)}"
            print(f"[Tool | Action] ✅ Action is performed successfully.")
        except Exception as error:
            print(f"[Tool | Action] ⛔️ Action Failed.")
            raise error

        return tool_result
    

    # ----------------------------------------------------------------------------
    # LLM Client
    # ----------------------------------------------------------------------------
    # create llm client
    openai_client = ChatOpenAI(**model_config)
    # bind tools to llm client
    openai_client_with_tools = openai_client.bind_tools([action_tool])


    # ----------------------------------------------------------------------------
    # Define Nodes
    # ----------------------------------------------------------------------------
    # define actor node
    def _actor(state: ReActState):
        # create request message list (insert system message in current message history).
        messages: Sequence[AnyMessage] = state['messages']
        request_messages: Sequence[AnyMessage] = [SystemMessage(content=REACT_SYSTEM_PROMPT)] + messages

        # get response from llm client with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try: 
                # get response
                response: AIMessage = openai_client_with_tools.invoke(request_messages)
                print(f"\n[Actor] ✅ Request succeed on attept {attempt+1}/{max_retries}")
                break
            except Exception as error:
                print(f"[Actor] ⚠️ Request failed on attept {attempt+1}/{max_retries}")
                # raise error when attempt hit max retry limit.
                if attempt + 1 == max_retries:
                    print(f"[Actor] ⛔️ Model Request failed. Please Try Later.")
                    raise error
        
        # get token usages.
        try:
            input_tokens = response.usage_metadata['input_tokens']
            output_tokens = response.usage_metadata['output_tokens']
            total_tokens = response.usage_metadata['total_tokens']
            print(f"[Actor] ✅ Token usage is collected successfully.")
        except Exception as error:
            print(f"[Actor] ⛔️ Response message doesn't contain token usage metadata.")
            raise error
        
        # update agent state
        return {
            'messages' : [response],
            'input_tokens' : input_tokens,
            'output_tokens' : output_tokens,
            'total_tokens' : total_tokens,
        }
    
    # define tool node
    def _tools(state: ReActState):
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
    

    # ----------------------------------------------------------------------------
    # Define Conditional edge function (to ToolNode)
    # ----------------------------------------------------------------------------
    def _should_continue(state:ReActState):
        messages = state['messages']
        last_msg: AIMessage = messages[-1]

        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            print("[Conditional Edge] 🌏 Actor make tool call.")
            return 'tools'
        else:
            idx = -2
            msg = None
            while not isinstance(msg, AIMessage): 
                idx -= 1
                msg = messages[idx]
            
            for tool_call in msg.tool_calls:
                if 'complete_task' in tool_call['args']['code']:
                    print("[Conditional Edge] ⭐️ Actor complete task.")
                    return 'end'
            
            print("[Conditional Edge] ⚠️ Actor didn't call `complete_task`.")
            return 'end'

    # ----------------------------------------------------------------------------
    # Define ReAct Workflow
    # ----------------------------------------------------------------------------
    # create graph builder
    workflow = StateGraph(state_schema=ReActState)

    # add nodes
    workflow.add_node("actor", _actor)               # actor
    workflow.add_node("tools", _tools)               # tool

    # add edges
    workflow.add_edge(START, "actor")
    workflow.add_conditional_edges(
        "actor",
        _should_continue,
        {
            'tools' : 'tools',
            'end' : END
        }
    )
    workflow.add_edge('tools', 'actor')

    # compile graph and return CompiledStateGraph instance
    return workflow.compile()