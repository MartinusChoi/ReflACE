from typing import Sequence, Any

from langchain_openai import ChatOpenAI
from langchain.messages import AnyMessage, AIMessage, SystemMessage

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from appworld import AppWorld

from ..prompt.react.system_prompt import REACT_SYSTEM_PROMPT
from ..state import State
from .base import BaseAgent


# --------------------------------------------------------------------------------------------------------
# ReAct Agent
# --------------------------------------------------------------------------------------------------------
class ReActAgent(BaseAgent):
    """
    ReAct Agent Class.
    """
    def __init__(
        self,
        env: AppWorld,
        system_prompt: str = REACT_SYSTEM_PROMPT,
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
    
    # ----------------------------------------------------------------------------
    # Define Actor Node
    # ----------------------------------------------------------------------------
    def _get_actor_node(self):
        
        # Actor Node
        # ============================================================================================================
        def _actor(state: State):
            # if actor called `complete_task()` before, stop without llm response
            for msg in reversed(state['messages']):
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if 'complete_task' in  tool_call['args']['code']:
                            return None

            # create request message list (insert system message in current message history)
            messages: Sequence[AnyMessage] = state['messages']
            request_messages: Sequence[AnyMessage] = [SystemMessage(content=self.system_prompt)] + messages

            # get response from llm client with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # get response
                    response: AIMessage = self.openai_client_with_tools.invoke(request_messages)
                    print(f"\n[Actor] ✅ Request succeed on attept {attempt+1}/{max_retries}")
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        print(f"[Actor] 🌏 Actor make tool call. > {len(response.tool_calls)} tool calls")
                        break
                    else:
                        print("[Actor] ⚠️ Actor didn't make tool call. Actor will retry.")
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
        # ============================================================================================================

        return _actor


    def _build_agent(self):
        # ----------------------------------------------------------------------------
        # Get Nodes
        # ----------------------------------------------------------------------------
        _actor = self._get_actor_node()
        _tools = self._get_tool_node()

        # ----------------------------------------------------------------------------
        # Get Conditional Edges
        # ----------------------------------------------------------------------------
        _should_continue = self._get_should_continue()

        # ----------------------------------------------------------------------------
        # Define ReAct Workflow
        # ----------------------------------------------------------------------------
        # create graph builder
        workflow = StateGraph(state_schema=State)

        # add nodes
        workflow.add_node("actor", _actor)   # actor
        workflow.add_node("tools", _tools)   # tool

        # add edges
        workflow.add_edge(START, "actor")
        workflow.add_conditional_edges(
            'actor',
            _should_continue,
            {
                'end' : END,
                'tools' : 'tools'
            }
        )
        workflow.add_edge("tools", "actor")

        # compile graph and return CompiledStateGraph instance
        return workflow.compile()
    
    
    def invoke(self, state: State):
        return self.agent.invoke(state)