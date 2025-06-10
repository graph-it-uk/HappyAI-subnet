import datetime
import uuid
from typing import List

from fastapi import Depends

from app.config.common_sys_prompts import sys_prompt_rethink_chat_name_if_needed, sys_prompt_chat_name_creation
from app.config.const import DEFAULT_GREETING
from app.dao.dynamo import DynamoDbDAO, get_dynamodb_dao
from app.dao.postgres import get_postgres_dao, PostgresDAO
from app.dto.common_dto import SenderInput, MessageInfo, SendMessageRequest, Sender
from app.models import Chat
from app.services.orchestrator import OrchestratorService, get_orchestrator_service


class ChatService:

    def __init__(self, postgres_dao: PostgresDAO, dynamodb_dao: DynamoDbDAO,
                 orchestrator_service: OrchestratorService):
        self.__postgres_dao = postgres_dao
        self.__dynamodb_dao = dynamodb_dao
        self.__orchestrator_service = orchestrator_service

    def add_chat(self, chat_info: Chat) -> Chat:
        chat = self.__postgres_dao.persist(chat_info)

        return chat

    def create_initial_message(self, user_id: uuid.uuid4, chat_id: uuid.uuid4, user_input: SenderInput) -> MessageInfo:
        message_info = MessageInfo(
            id=uuid.uuid4(),
            text=user_input.text,
            sender=user_input.sender,
            q_type='1'
        )

        self.__dynamodb_dao.create_initial_message(user_id, chat_id, message_info)
        return message_info

    def get_chat(self, user_id, chat_id: str) -> Chat:
        chat = self.__postgres_dao.get_chat_info(user_id, chat_id)

        if chat is None:
            raise Exception('Unexpected chat issue')

        return chat

    def update_chat_name_and_llm_name(self, user_id: str, chat_id: str, session_name: str = None, model: str = 'core'):
        self.__postgres_dao.update_chat_name_and_llm_name(user_id, chat_id, session_name, model)

    def is_chat_exist(self, chat_id: str) -> bool:
        chat_exists = self.__postgres_dao.is_chat_exist(chat_id)

        return chat_exists

    def is_messages_exist(self, user_id: str, chat_id: str) -> bool:
        return self.__dynamodb_dao.record_exists(user_id, chat_id)

    def persist_message_to_dynamodb(self, user_id, chat_id, sender_input: SenderInput, q_type: int = None) -> MessageInfo:
        new_message = MessageInfo(
            id=str(uuid.uuid4()),
            text=sender_input.text,
            sender=sender_input.sender,
            q_type=q_type,
            sent_at=datetime.datetime.now()
        )

        self.__dynamodb_dao.append_message(user_id, chat_id, new_message)

        return new_message

    def get_chat_messages(self, user_id: str, chat_id: str) -> list[MessageInfo]:
        return self.__dynamodb_dao.get_chat_messages(user_id, chat_id)

    def get_chat_last_messages(self, N: int, chat_id: str, user_id: str) -> list[MessageInfo]:
        chat_messages_response: list[MessageInfo] = self.get_chat_messages(chat_id, user_id)

        if chat_messages_response:
            last_messages_info = chat_messages_response[-N:]
            return last_messages_info

        return []

    def get_last_messages_as_string(self, N: int, user_id: str, chat_id: str) -> list[MessageInfo]:
        return self.get_chat_last_messages(N, user_id, chat_id)

    def format_messages_list_to_str(self, messages: list[MessageInfo]) -> str:
        return "\n\n".join(m.sender + ': ' + m.text for m in messages)

    def list_chats(self, user_id: str) -> list[Chat]:
        return self.__postgres_dao.list_chats_by_user_id(user_id)

    def delete_chat_info(self, chat_id: str, user_id: str):
        return self.__postgres_dao.delete_chat_info(chat_id, user_id)

    def create_chat(self, user_id, user_name: str, recommendation_from_mood_journal: str) -> tuple[Chat, MessageInfo]:
        chat_id = uuid.uuid4()

        sender_input = SenderInput(
            sender=Sender.ASSISTANT,
            text=DEFAULT_GREETING if recommendation_from_mood_journal is None else recommendation_from_mood_journal
        )

        chat = self.add_chat(
            Chat(
                id=chat_id,
                fk_user_id=user_id,
                started_at=datetime.datetime.now()
            )
        )

        message_info = self.create_initial_message(user_id, chat.id, sender_input)

        return chat, message_info

    def process_chat_message(self, user_id: str, send_message_request: SendMessageRequest, assistant_input: SenderInput, q_type) -> MessageInfo:

        _ = self.persist_message_to_dynamodb(user_id, send_message_request.chat_id, send_message_request.user_input, q_type)
        assistant_message_info = self.persist_message_to_dynamodb(user_id, send_message_request.chat_id, assistant_input, q_type)

        return assistant_message_info

    def get_chat_name(self, chat: Chat, session_input, next_q_type, is_issue):
        if chat.session_name is None and is_issue:
            session_name = self.__orchestrator_service.get_session_name(session_input, sys_prompt_chat_name_creation)

        elif next_q_type['question_flag'] == 2 and chat.session_name is not None:
            rethink_prompt_filled = sys_prompt_rethink_chat_name_if_needed.format(
                conversation_content=session_input,
                initial_chat_name=chat.session_name
            )
            session_name = self.__orchestrator_service.get_session_name(session_input, rethink_prompt_filled)

        else:
            session_name = chat.session_name

        return session_name


def get_chat_service(postgres_dao: PostgresDAO = Depends(get_postgres_dao),
                     dynamodb_dao: DynamoDbDAO = Depends(get_dynamodb_dao),
                     orchestrator_service: OrchestratorService = Depends(get_orchestrator_service)) -> ChatService:
    return ChatService(postgres_dao, dynamodb_dao, orchestrator_service)



