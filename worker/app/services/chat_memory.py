from app.dao.dynamo import DynamoDbDAO, get_dynamodb_dao

from fastapi import Depends

from app.services.llm_response import LlmResponseService


class ChatMemorService:
    def __init__(self, dynamodb_dao: DynamoDbDAO):
        self.__dynamodb_dao = dynamodb_dao

    def add_chat_memory(self, user_id, chat_id, memory_text):
        self.__dynamodb_dao.add_chat_memory(user_id, chat_id, memory_text)

    def update_chat_memory(self, user_id, chat_id, memory_text):
        self.__dynamodb_dao.update_chat_memory(user_id, chat_id, memory_text)


def get_llm_response_service(dynamodb_dao: DynamoDbDAO = Depends(get_dynamodb_dao)):
    return LlmResponseService(dynamodb_dao)
