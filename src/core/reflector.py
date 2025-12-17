from typing import List, Dict

class Reflector:
    """
    Reflector agent: Analyzes an episode (trajectory) to generate general insights.
    """
    def __init__(self, llm_client):
        self.llm = llm_client

    def reflect_on_episode(self, task: str, history: List[Dict[str, str]], success: bool) -> List[str]:
        """
        Analyze the episode and return a list of insights (rules/strategies).
        """
        prompt = f"""
        Analyze the following agent trajectory for the task: "{task}".
        Did the agent succeed? {success}.
        
        Extract 1-3 general rules or strategies that would help an agent solve this or similar tasks more efficiently in the future.
        Do not just describe what happened. Formulate them as imperative "Playbook" rules.
        
        Trajectory:
        {str(history)}
        
        Format output as a bulleted list.
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.chat_completion(messages)
        
        # Parse bullets
        insights = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("- "):
                insights.append(line[2:])
            elif line.startswith("* "):
                insights.append(line[2:])
                
        return insights
