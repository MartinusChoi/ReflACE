from .base import BaseAgent
from .react import ReActAgent
from typing import Dict, Any, List

class ReflexionAgent(BaseAgent):
    """
    Reflexion Agent wrapper.
    It runs the inner ReAct agent. If it fails, it reflects and retries.
    """
    def __init__(self, llm_client, env, max_retries: int = 2):
        super().__init__(llm_client, env)
        self.react_agent = ReActAgent(llm_client, env)
        self.max_retries = max_retries
        self.long_term_memory: List[str] = [] # List of reflections

    def _reflect(self, history: List[Dict[str, str]], task: str) -> str:
        """
        Generate reflection based on failure history.
        """
        # Simple reflection prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Analyze the following trace and explain why it failed. Then provide a plan to avoid this mistake."},
            {"role": "user", "content": f"Task: {task}\nTrace:\n{str(history)}"}
        ]
        response = self.llm.chat_completion(messages)
        return response

    def run(self, task: str, max_steps: int = 10) -> Dict[str, Any]:
        final_success = False
        
        for trial in range(self.max_retries + 1):
            # Inject memory into ReAct agent prompt
            # For simplicity, we append memory to the task description or system prompt of the sub-agent
            # But the ReActAgent needs to handle this.
            # Let's modify ReAct logic slightly to accept 'memory' or we prepend it to task.
            
            task_with_memory = task
            if self.long_term_memory:
                 task_with_memory += "\n\nTips from previous attempts:\n" + "\n".join(self.long_term_memory)
            
            print(f"Trial {trial} START")
            result = self.react_agent.run(task_with_memory, max_steps)
            
            if result['success']:
                final_success = True
                break
            else:
                # Failed, create reflection
                print(f"Trial {trial} FAILED. Reflecting...")
                reflection = self._reflect(result['history'], task)
                self.long_term_memory.append(reflection)
                self.react_agent.reset() # Important: reset state for next trial
                
        return {"success": final_success, "trials": trial + 1}
        
    def reset(self):
        super().reset()
        self.long_term_memory = []
        self.react_agent.reset()
