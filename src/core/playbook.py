from typing import Dict, Any, List
import numpy as np

from langchain_openai import OpenAIEmbeddings


def cosine_similarity(
    vec1:List[float], 
    vec2:List[float]
) -> float:
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # 코사인 유사도 = (A · B) / (||A|| * ||B||)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    return dot_product / (norm_vec1 * norm_vec2)


class PlayBook:
    def __init__(
        self,
    ) -> None:
        self.playbook: Dict[str, List[Dict[str, Any]]] = {
            'STRATEGIES AND HARD RULES' : [],
            'USEFUL CODE SNIPPETS AND TEMPLATES' : [],
            'TROUBLESHOOTING AND PITFALLS' : []
        }
        self.embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
    
    def _get_embedding(
        self, 
        content: str
    ) -> List[float]:
        return self.embedding_model.embed_query(content)
    
    def add_to_playbook(
        self,
        section: str,
        content: str
    ) -> None:
        embedding: List[float] = self._get_embedding(content)

        for bullet in self.playbook[section]:
            if cosine_similarity(embedding, bullet['embedding']) >= 0.8:
                self.playbook[section]['count'] += 1
                return
        
        self.playbook[section].append({
            'content' : content,
            'embedding' : embedding,
            'count' : 0
        })
    
    def to_str(self):
        playbook = ""

        for section_title, section_body in self.playbook.items():
            playbook += f"{section_title}:"
            for i, bullet in enumerate(section_body):
                if section_title == 'STRATEGIES AND HARD RULES':
                    playbook += f"  * shr-{i:05d} : {bullet['content']}\n"
                if section_title == 'USEFUL CODE SNIPPETS AND TEMPLATES':
                    playbook += f"  * code-{i:05d} : {bullet['content']}\n"
                if section_title == 'TROUBLESHOOTING AND PITFALLS':
                    playbook += f"  * ts-{i:05d} : {bullet['content']}\n"

            playbook += "\n"
        
        return playbook