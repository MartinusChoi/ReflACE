from .playbook import Playbook
from typing import List

class Curator:
    """
    Curator agent: Manages the evolution of the Playbook.
    """
    def __init__(self, llm_client, playbook: Playbook):
        self.llm = llm_client
        self.playbook = playbook

    def update(self, new_insights: List[str]):
        """
        Integrate new insights into the playbook.
        Performs semantic deduplication.
        """
        # For this simplified version, we just check exact string set membership
        # or use LLM to check if insight is already covered.
        
        current_content = self.playbook.get_content()
        
        for insight in new_insights:
             # Logic to check if 'insight' is already semantically present in 'current_content'
             # Here we use a simpler heuristic for prototype
             if insight not in current_content:
                 self.playbook.add_insight(insight)
