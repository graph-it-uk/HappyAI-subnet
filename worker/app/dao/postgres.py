from typing import Type
import uuid
from sqlalchemy import and_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import Chat, Questionnaire, Base, Mood
from app.utils.sql_connector import get_postgres_db, engine


class PostgresDAO:
    Base.metadata.create_all(engine, checkfirst=True)
    __db: Session = get_postgres_db()

    def persist(self, obj: Chat):
        try:
            self.__db.add(obj)
            self.__db.commit()

            return obj
        except Exception as e:
            self.__db.rollback()
            print(f"Error: {e}")

    def get_chat_info(self, user_id: str, chat_id) -> Chat:
        chat_info = self.__db.query(Chat).filter(Chat.id == chat_id and Chat.fk_user_id == user_id).one_or_none()
        if chat_info:
            return chat_info
        else:
            raise Exception("Chat does not exist")

    def delete_chat_info(self, chat_id: str, user_id: str) -> str:
        try:
            chat = self.__db.query(Chat).filter(and_(Chat.id == chat_id, Chat.fk_user_id == user_id)).first()

            if chat:
                self.__db.delete(chat)
                self.__db.commit()
                return 'Success'
            else:
                print("Chat not found.")
                return 'Chat not found'

        except SQLAlchemyError as e:
            # Rollback the session in case of an error
            self.__db.rollback()
            print(f"An error occurred: {e}")
            return 'Error occurred while deleting chat'

    def update_chat_name_and_llm_name(self, user_id: str, chat_id: str, session_name: str = None,
                                      llm_model: str = 'core'):
        chat = self.get_chat_info(user_id, chat_id)

        chat.session_name = session_name
        chat.llm_model = llm_model
        try:
            self.__db.commit()
        except Exception as e:
            self.__db.rollback()
            print(f"Error occurred: {e}")

    def is_chat_exist(self, chat_id: uuid.uuid4) -> bool:
        return self.__db.query(Chat).filter(Chat.id == chat_id).count() > 0

    def list_chats_by_user_id(self, user_id) -> list[Type[Chat]]:

        chat_list = self.__db.query(Chat).filter(Chat.fk_user_id == user_id).all()
        return chat_list

    def get_questionnaire(self, user_id) -> Type[Questionnaire]:

        questionnaire = self.__db.query(Questionnaire).filter(Questionnaire.user_id == user_id).one_or_none()

        return questionnaire

    def get_mood_journal(self, user_id):
        mood_journal = self.__db.query(Mood).filter(Mood.user_id == user_id).order_by(Mood.created_at.desc()).first()

        return mood_journal


def get_postgres_dao():
    return PostgresDAO()
