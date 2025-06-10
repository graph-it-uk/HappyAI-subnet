
import enum
from enum import Enum
from typing import Dict, List, Optional

import bittensor as bt
import pydantic


class Version(pydantic.BaseModel):
    major: int
    minor: int
    patch: int


class Role(str, enum.Enum):
    """Message is sent by which role?"""

    user = "user"
    assistant = "assistant"
    system = "system"

class Message(pydantic.BaseModel):
    role: Role
    content: str


class CompletionSynapse(bt.Synapse):
    """
    """
    request_id: int
    messages: list[Message]
    user_input: str

    version: Optional[Version] = None

    results: Optional[str] = None

    def deserialize(self) -> List[Dict]:
        return self.results
