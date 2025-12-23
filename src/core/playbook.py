from typing import List, Dict

class Playbook:
    """
    Manages the long-term memory (Playbook) for the ACE agent.
    """
    def __init__(self):
        self.insights: Dict[str, Dict[str, str]] = {}
        self.insight_ids: List[str] = None

    def add_insight(self, insight: str):
        """Add a new insight to the playbook."""
        if insight not in self.insights:
            self.insights.append(insight)

    def get_content(self) -> str:
        """Return the formatted playbook content for injection into prompts."""
        if not self.insights:
            return ""
        
        content = "## Playbook\n"
        for i, insight in enumerate(self.insights):
            content += f"- {insight}\n"
        return content
