from typing import List, Optional

from pydantic import BaseModel


class Metadata(BaseModel):
    chat_id: str
    chat_name: str
    chat_summary: str
    user_id: str


class RagChat(BaseModel):
    id: str
    metadata: Metadata
    score: float
    values: Optional[List[str]] = []
