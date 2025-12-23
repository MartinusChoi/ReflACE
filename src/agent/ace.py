from .base import BaseAgent
from .react import ReActAgent
from ..env.appworld_env import AppWorldEnv
from ..llm.openai_client import OpenAIClient
from ..core.playbook import Playbook
from ..core.trajectory import Trajectory
from typing import Dict, Any

PROMPT = {
    'only_ace' : """
**Task:**
{instruction}

**Model Action Trajectory:**
{trajectory}

**Part of Playbook that's used by the generator to answer the question:**
{playbook}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}
""",
    'with_reflexion' : """
"""
}

class ACEAgent(BaseAgent):
    """
    ACE Agent wrapper.
    Updating the Playbook after episodes.
    """
    def __init__(
        self,
        actor_client:OpenAIClient,
        reflector_client:OpenAIClient,
        env:AppWorldEnv,
    ):
        super().__init__(
            actor_client=actor_client    
        )
        self.playbook = Playbook()
        self.reflector_client = reflector_client
        self.actor = ReActAgent(actor_client=self.actor_client)
    
    def _build_reflect_prompt(
        self,
        instruction:str,
        trajectory:Trajectory,
        playbook:Playbook
    ) -> str:
        return PROMPT['only_ace'].format(
            instruction=instruction,
            trajectory=trajectory.get_content(),
            playbook=playbook.get_content()
        )


    def _reflector(
        self,
        env:AppWorldEnv,
        trajectory:Trajectory
    ) -> str:

        pass


    def run(self, task: str, max_steps: int = 10) -> Dict[str, Any]:
        # 1. Inject Playbook
        current_playbook = self.playbook.get_content()
        
        # We need to pass the playbook to the inner agent. 
        # Since ReAct/Reflexion don't explicitly take it in run(), 
        # we might need to set it on the instance or prepend to task.
        # Ideally, we set it on the underlying ReAct agent.
        
        if self.use_reflexion:
            self.inner_agent.react_agent.playbook = current_playbook
        else:
            self.inner_agent.playbook = current_playbook
            
        # 2. Run Episode
        result = self.inner_agent.run(task, max_steps)
        
        # 3. Slow Loop: Reflect & Curate (Training Phase)
        # In a real evaluation, we might freeze the playbook, but here we update it.
        history = result.get('history', []) 
        if self.use_reflexion and 'trials' in result:
             # For Reflexion, we might want to gather history from all trials or just the last?
             # For simplicity, let's assume the inner agent's last state or we need to capture full trace.
             # The simple Reflexion implementation I wrote resets history. 
             # We might need to improve ReflexionAgent to return full combined history.
             pass
        
        print("ACE: Reflecting on episode...")
        insights = self.reflector.reflect_on_episode(task, history, result['success'])
        
        print(f"ACE: Generated {len(insights)} insights. Curating...")
        self.curator.update(insights)
        
        return result

    def reset(self):
        super().reset()
        self.inner_agent.reset()
        # Note: Playbook is NOT reset between episodes! That's the point.
