from .base import BaseAgent
from typing import Dict, Any, List

class ReActAgent(BaseAgent):
    """
    Standard ReAct Agent.
    """
    def __init__(self, llm_client, env):
        super().__init__(llm_client, env)
        
        # ACE playbook to be injected if needed
        self.playbook = "" 

    def _build_prompt(self, obs: str, task: str) -> List[Dict[str, str]]:
        messages = [
            {
                "role": "system", 
                "content":
                """
                You are a helpful agent. 
                Solve the task using ReAct format: Thought, Action. 
                Available actions: eat, look.
                """
            }
        ]
        
        # Inject Playbook if available (ACE Setting)
        if self.playbook:
            messages.append(
                {
                    "role": "system", 
                    "content": f"Here is a Playbook of strategies to help you:\n{self.playbook}"
                }
            )

        # trajectory
        history_text = f"Task: {task}\n"
        for item in self.history:
            history_text += f"\nObs: {item['obs']}\n"
            history_text += f"Thought: {item['thought']}\n"
            history_text += f"Action: {item['action']}\n"
        
        history_text += f"\nObs: {obs}\n"
        
        messages.append({"role": "user", "content": history_text})
        return messages

    def _parse_response(self, response: str) -> Dict[str, str]:
        # Simple parsing logic (robustness needs improvement for real LLM)
        thought = ""
        action = ""
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
        
        return {"thought": thought, "action": action}

    def run(self, task: str, max_steps: int = 10) -> Dict[str, Any]:
        obs, info = self.env.reset()
        success = False
        
        for step in range(max_steps):
            messages = self._build_prompt(obs, task)
            response = self.llm.chat_completion(messages, stop=["Obs:"])
            parsed = self._parse_response(response)
            
            action = parsed['action']
            if not action:
                # Fallback if parsing fails or LLM outputs nothing
                action = "look" 

            new_obs, reward, done, truncated, info = self.env.step(action)
            
            self.history.append({
                "obs": obs,
                "thought": parsed['thought'],
                "action": action
            })
            
            obs = new_obs
            if done:
                success = True
                break
                
        return {"success": success, "history": self.history, "steps": step + 1}

    def reset(self):
        super().reset()
