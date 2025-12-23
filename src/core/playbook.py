from typing import List, Dict
from uuid import uuid4

class Playbook:
    """
    Manages the long-term memory (Playbook) for the ACE agent.
    """
    def __init__(self):
        self.insights: Dict[str, Dict[str, Dict[str, str]]] = {}

    def get_ids(self) -> List[str]:
        return list(self.insights.keys())

    def add_insight(
        self,
        insight: str
    ):
        """
        Add a new insight to the playbook.
        
        Args:
            insight (str): The insight to add.
        """
        if insight not in self.insights:
            self.insights[str(uuid4())] = {
                'tag' : 'neutral',
                'insight' : insight
            }
        
    def update_tag(
        self,
        insight_id:str,
        tag: str
    ):
        """
        Update the tag of an existing insight in the playbook.
        
        Args:
            insight_id (str): The ID of the insight to update.
            tag (str): The updated tag.
        """
        if insight_id in self.insights:
            self.insights[insight_id]['tag'] = tag
        else:
            raise ValueError(f"Insight ID {insight_id} not found in playbook.")
    
    def update_insight(
        self,
        insight_id:str,
        insight:str
    ):
        """
        Update an existing insight in the playbook.
        
        Args:
            insight_id (str): The ID of the insight to update.
            insight (str): The updated insight.
        """
        if insight_id in self.insights:
            self.insights[insight_id]['insight'] = insight
        else:
            raise ValueError(f"Insight ID {insight_id} not found in playbook.")

    def to_playbook(self) -> str:
        """
        Return the formatted playbook content for injection into prompts.

        Returns:
            str: The formatted playbook content.
        """
        
        content = "<Playbook>\n"
        for i, insight in enumerate(self.insights):
            content += f"- {insight['tag']} : {insight['insight']}\n"
        content += "</Playbook>\n\n"

        return content
