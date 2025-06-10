from fastapi import Depends

from app.dao import rag
from app.dao.dynamo import get_dynamodb_dao
from app.dto.rag import RagChat
from app.models import Chat
from app.services.llm_response import get_llm_response_service
from app.utils.secrets import get_open_ai_client, get_pinecone_client


class RagService:
    def __init__(self, open_ai_client, pinecone_client, dynamodb_dao, llm_response_service):
        self.__rag_dao = rag.get_rag_dao(open_ai_client, pinecone_client)
        self.__dynamodb_dao = dynamodb_dao
        self.__llm_response_service = llm_response_service

    def prepare_and_upsert_vectors(self, chat: Chat, chat_summary: str) -> list:
        vectors = []
        chat_id_str = str(chat.id)
        user_id_str = str(chat.fk_user_id)
        embedding = self.__rag_dao.get_embedding(chat_summary)

        # Prepare vector with chat_summary and user_id in metadata
        vector = (
            chat_id_str,
            embedding,
            {
                "user_id": user_id_str,  # Add user_id to metadata
                "chat_name": chat.session_name,
                "chat_id": chat_id_str,
                "chat_summary": chat_summary
            }
        )
        vectors.append(vector)

        self.__rag_dao.upsert_vectors(vectors)

        return vectors

    def query_index_with_filter(self, query_text, user_id, top_k=2, score_threshold=0.3) -> list[RagChat]:
        result = self.__rag_dao.query_index_with_filter(query_text, user_id, top_k, score_threshold)

        result_dicts = [
            {
                'id': scored_vector.id,
                'metadata': scored_vector.metadata,
                'score': scored_vector.score,
                'values': scored_vector.values
            }
            for scored_vector in result
        ]

        return [RagChat(**r) for r in result_dicts]

    def process_chat_summary(self, chat, next_q_type):
        if next_q_type['question_flag'] == 35 and chat.session_name is not None:
            chat_summary = self.__llm_response_service.summarize_chat(chat.fk_user_id, str(chat.id))
            self.prepare_and_upsert_vectors(chat, chat_summary)

            return chat_summary
        else:
            return None


def get_rag_service(open_ai_client=Depends(get_open_ai_client), pinecone_client=Depends(get_pinecone_client),
                    dynamodb_dao=Depends(get_dynamodb_dao), llm_response_service=Depends(get_llm_response_service)):
    return RagService(open_ai_client, pinecone_client, dynamodb_dao, llm_response_service)
