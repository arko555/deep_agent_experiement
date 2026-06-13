import os
import json
from typing import Dict, List, Optional
from datetime import datetime

class MemoryManager:
    def __init__(self, workspace_root: str = "./workspace"):
        self.workspace_root = workspace_root
        self.memory_dir = os.path.join(self.workspace_root, ".memory")
        os.makedirs(self.memory_dir, exist_ok=True)

    def _get_thread_path(self, thread_id: str) -> str:
        return os.path.join(self.memory_dir, f"thread_{thread_id}.json")

    def _get_entity_path(self, entity_id: str) -> str:
        return os.path.join(self.memory_dir, f"entity_{entity_id}.json")

    def save_thread_memory(self, thread_id: str, messages: List[Dict]) -> None:
        path = self._get_thread_path(thread_id)
        with open(path, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "messages": messages}, f)

    def get_thread_memory(self, thread_id: str) -> List[Dict]:
        path = self._get_thread_path(thread_id)
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("messages", [])

    def save_entity_memory(self, entity_id: str, data: Dict) -> None:
        path = self._get_entity_path(entity_id)
        with open(path, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f)

    def get_entity_memory(self, entity_id: str) -> Optional[Dict]:
        path = self._get_entity_path(entity_id)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("data")
