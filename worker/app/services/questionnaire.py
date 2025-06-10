from typing import List, Type

from fastapi import Depends
from sqlalchemy.orm import InstrumentedAttribute

from app.dao.dynamo import DynamoDbDAO, get_dynamodb_dao
from app.dao.postgres import PostgresDAO, get_postgres_dao
from app.dto.common_dto import Question
from app.models import Questionnaire
from app.services.llm_response import get_llm_response_service, LlmResponseService


class QuestionnaireService:

    def __init__(self, postgres_dao: PostgresDAO, dynamodb_dao: DynamoDbDAO, llm_response_service: LlmResponseService):
        self.__postgres_dao = postgres_dao
        self.__dynamodb_dao = dynamodb_dao
        self.__llm_response_service = llm_response_service

    def get_questions(self, user_id: str):
        questionnaire: Type[Questionnaire] = self.__postgres_dao.get_questionnaire(user_id)

        return questionnaire.data if questionnaire else None


def get_questionnaire_service(postgres_dao: PostgresDAO = Depends(get_postgres_dao),
                              dynamodb_dao: DynamoDbDAO = Depends(get_dynamodb_dao),
                              llm_response_service=Depends(get_llm_response_service)):
    return QuestionnaireService(postgres_dao, dynamodb_dao, llm_response_service)
