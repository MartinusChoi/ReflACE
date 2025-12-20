from typing import Dict, Any, List, TypedDict, Annotated, Literal
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from .base import BaseAgent
from .react import ReActAgent
from ..llm.openai_client import OpenAIClient
from ..env.appworld_env import AppWorldEnv

# -------------------------------------------------------------------------------------
# Graph State Schema
# -------------------------------------------------------------------------------------
class ReflexionState(TypedDict):
    """
    Represents the state of the Reflexion agent.
    """
    user_task: str
    trial_num: int
    trajectory: List[str] # Detailed history of steps/actions
    reflections: List[str] # List of critiques/feedback
    is_success: bool
    final_response: str # Final answer or result summary


# -------------------------------------------------------------------------------------
# Response Schemas for Nodes
# -------------------------------------------------------------------------------------
class ReflectionResponse(BaseModel):
    error: str = Field(description="Error message.")
    
class ReflectionOutput(BaseModel):
    reflection: str = Field(description="Combined reflection and advice for the next trial.")

# -------------------------------------------------------------------------------------
# Main Reflexion Agent Class
# -------------------------------------------------------------------------------------
class ReflexionAgent(BaseAgent):
    """
    Reflexion Agent that orchestrates a ReActAgent with a Reflection loop using LangGraph.
    """

    def __init__(self, llm_client: OpenAIClient):
        super().__init__(llm_client)
        # We instantiate ReActAgent on demand or keep a reference if lightweight.
        # Since ReActAgent holds no persistent state between runs (it initializes fresh trajectory),
        # we can instantiate it once.
        self.actor_agent = ReActAgent(llm_client)
        self.max_retries = 3
        self.app_graph = self._build_graph()

    def _build_graph(self):
        """
        Constructs the LangGraph state machine.
        """
        graph = StateGraph(ReflexionState)

        graph.add_node("actor", self._actor_node)
        graph.add_node("evaluator", self._evaluator_node)
        graph.add_node("reflector", self._reflector_node)

        graph.add_edge(START, "actor")
        graph.add_conditional_edges(
            "actor",
            self._should_continue,
            {
                "evaluate": "evaluator",
                "end": END
            }
        )
        graph.add_conditional_edges(
            "evaluator",
            self._evaluate_decision,
            {
                "reflect": "reflector",
                "end": END
            }
        )
        graph.add_edge("reflector", "actor")

        return graph.compile()
    
    # -------------------------------------------------------------------------------------
    # Nodes
    # -------------------------------------------------------------------------------------
    def _actor_node(self, state: ReflexionState):
        """
        Executes the ReAct agent.
        """
        task_prompt = state['user_task']
        trial_num = state.get('trial_num', 0)
        reflections = state.get('reflections', [])
        
        # Augment task with memory/reflections
        if reflections:
            reflection_text = "\n".join(reflections)
            task_prompt += f"\n\nPrevious attempts failed. Here are some reflections and tips to improve:\n{reflection_text}"
        
        # We need the Env to run the ReActAgent. 
        # CAUTION: The standard ReActAgent.run() signature is run(env, max_steps).
        # We need 'env' passed via config or we need to assume it's set somewhere.
        # In this architecture, usually 'env' is part of the graph config or passed in context.
        # But 'ReActAgent.run' is blocking and runs the whole loop. 
        
        # We will retrieve 'env' from the state or config. 
        # Limitation: BaseAgent doesn't store env by default in this codebase structure (checked react.py).
        # We'll assume the caller passes 'env' in the inputs, but StateGraph inputs must match Schema.
        # We will rely on a hack: we will attach 'env' to the agent instance for the duration of the run,
        # OR we assume 'env' is accessible.
        
        # Let's check how 'run' is called. 
        # For now, I will use a placeholder for env and rely on the fact that 'run' method of attributes 
        # needs the env.
        # I'll modify the 'run' method of ReflexionAgent to accept 'env' and store it temporarily.
        
        env = self._current_env # Expect this to be set in .run()
        
        print(f"--- Actor Loop (Trial {trial_num}) ---")
        
        # Run ReAct Agent
        # ReActAgent.run returns Dict[str, Any]
        result = self.actor_agent.run(env, max_steps=30)
        
        trajectory_obj = result.get('trajectory')
        success = result.get('success', False)
        
        # Extract meaningful trajectory text for reflection
        # Trajectory object has .to_context() or similar.
        trajectory_str = str(trajectory_obj.to_context()) if trajectory_obj else "No trajectory"

        return {
            "trajectory": state.get('trajectory', []) + [trajectory_str],
            "is_success": success,
            "trial_num": trial_num + 1,
            "final_response": "Success" if success else "Failed"
        }

    def _evaluator_node(self, state: ReflexionState):
        """
        Evaluates the result.
        """
        # In this specific setup, ReActAgent already returns 'success' boolean based on its own logic (e.g. if it outputted AIMessage).
        # So we trust the Actor's own success flag initially.
        # However, we can add an extra layer of LLM evaluation if needed.
        # For now, we pass through the success status but format it for the Reflector.
        
        is_success = state['is_success']
        print(f"--- Evaluator: Success={is_success} ---")
        return {} # State already has 'is_success'
        
    def _reflector_node(self, state: ReflexionState):
        """
        Reflects on the failure.
        """
        print("--- Reflecting ---")
        
        task = state['user_task']
        last_trajectory = state['trajectory'][-1]
        
        # Create reflection prompt
        # We use the 'llm_client' specifically for this.
        # We need to construct a prompt for the specific LLM capabilities.
        
        prompt = f"""You are an advanced reasoning agent. You are given a task and a history of a failed attempt to solve it.
Your goal is to analyze the failure and provide concise, constructive feedback (reflection) to help the agent succeed in the next attempt.

Task: {task}

Failed Attempt Trajectory:
{last_trajectory}

Analyze the trajectory. keymissing strategies, or superfluous steps.
Provide your response as a concise reflection string.
"""
        messages = [UserMessage(content=prompt)]
        
        # Using self.llm (OpenAIClient)
        # OpenAIClient.get_response returns ChatMessageList or similar.
        response = self.llm.get_response(messages)
        
        reflection_text = "Reflexion failed to generate."
        if hasattr(response, 'messages') and len(response.messages) > 0:
            reflection_text = response.messages[-1].content
        
        return {
            "reflections": state.get('reflections', []) + [reflection_text]
        }

    # -------------------------------------------------------------------------------------
    # Edges
    # -------------------------------------------------------------------------------------
    def _should_continue(self, state: ReflexionState) -> Literal["evaluate", "end"]:
        """
        Determines if we should proceed to evaluation or end immediately (if success is obvious during actor).
        Actually, we always go to evaluator to check/finalize, or if we want to bypass evaluator on success, we can.
        """
        return "evaluate"

    def _evaluate_decision(self, state: ReflexionState) -> Literal["reflect", "end"]:
        if state['is_success']:
            return "end"
        
        if state['trial_num'] > self.max_retries:
            return "end"
            
        return "reflect"

    # -------------------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------------------
    def run(self, env: AppWorldEnv, task_description: str = None) -> Dict[str, Any]:
        """
        Entry point for Reflexion Agent.
        """
        self._current_env = env # Hack to pass env to graph nodes
        
        if task_description is None:
            # Try to extract from env
             task_description = env.get_instruction()

        initial_state = ReflexionState(
            user_task=task_description,
            trial_num=0,
            trajectory=[],
            reflections=[],
            is_success=False,
            final_response=""
        )
        
        final_state = self.app_graph.invoke(initial_state)
        
        return {
            "success": final_state['is_success'],
            "trajectory": final_state['trajectory'], # This is a list of trajectories
            "reflections": final_state['reflections']
        }
