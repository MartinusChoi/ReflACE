from .base import BaseAgent
from .react import ReActAgent
from .reflexion import ReflexionAgent
from ..core.playbook import Playbook
from ..core.curator import Curator
from ..core.reflector import Reflector
from typing import Dict, Any

class ACEAgent(BaseAgent):
    """
    ACE Agent wrapper.
    It manages the 'Slow Loop': Updating the Playbook after episodes.
    """
    def __init__(self, llm_client, env, use_reflexion: bool = False):
        super().__init__(llm_client, env)
        self.playbook = Playbook()
        self.reflector = Reflector(llm_client)
        self.curator = Curator(llm_client, self.playbook)
        
        self.use_reflexion = use_reflexion
        if use_reflexion:
            self.inner_agent = ReflexionAgent(llm_client, env)
        else:
            self.inner_agent = ReActAgent(llm_client, env)

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
