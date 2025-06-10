import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class Sender(str, Enum):
    USER = 'USER'
    ASSISTANT = 'ASSISTANT'


class Message(BaseModel):
    user_input: str
    assistant: str
    timestamp: datetime

    def to_string(self) -> str:
        return f'User: "{self.user_input}"\nAssistant: "{self.assistant}"'


class SenderInput(BaseModel):
    sender: Sender
    text: str

    def to_dict(self) -> dict:
        return {
            'sender': self.sender,
            'text': self.text,
        }


class DialogMessage(BaseModel):
    sender: str
    text: str

class DialogRequest(BaseModel):
    request_id: int
    user_input: DialogMessage
    dialog: List[DialogMessage]
    chat_llm_model: str


class DialogResponse(BaseModel):
    request_id: int
    q_type: Optional[int]
    assistant_message: str


class SendMessageRequest(BaseModel):
    user_input: SenderInput
    chat_id: Optional[str] = None


class UserIdRequest(BaseModel):
    user_id: uuid.UUID


class ChatIdRequest(BaseModel):
    chat_id: str


class GetChatMessageRequest(BaseModel):
    user_id: uuid.UUID
    chat_id: str


class MessageInfo(BaseModel):
    id: uuid.UUID
    text: str
    sender: Sender
    q_type: Optional[int]
    sent_at: datetime = datetime.now()

    def to_dict(self) -> dict:
        return {
            'id': str(self.id),
            'text': self.text,
            'sender': self.sender,
            'q_type': self.q_type,
            'sent_at': str(self.sent_at),
        }

    def to_string(self) -> str:
        return f'{self.sender.value}: "{self.text}"'


class ChatMessagesResponse(BaseModel):
    messages: List[MessageInfo]


class ChatInfoResponse(BaseModel):
    id: uuid.UUID

    fk_user_id: uuid.UUID

    started_at: datetime
    session_name: Optional[str]
    session_tag: Optional[str]

    class Config:
        from_attributes = True


class SendMessageResponse(BaseModel):
    chat: ChatInfoResponse
    assistant_message: MessageInfo


class CreateChatResponse(BaseModel):
    chat: ChatInfoResponse
    messages: List[MessageInfo]


class CreateMessageRequest(BaseModel):
    user_name: Optional[str] = None
    from_mood_journal: Optional[bool] = None


class Question(BaseModel):
    answer: str
    question: str
    answer_extra: str

    def to_dict(self) -> dict:
        return {
            'answer': self.answer,
            'question': self.question,
            'answer_extra': self.answer_extra
        }

    def to_str(self) -> str:
        return f"""
        {{
            "answer": "{self.answer}",
            "question": "{self.question}",
            "answer_extra": "{self.answer_extra}"
        }}"""
