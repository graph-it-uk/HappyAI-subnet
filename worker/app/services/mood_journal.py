from datetime import datetime

from fastapi import Depends

from app.config.common_sys_prompts import sys_prompt_recommendation
from app.dao.postgres import get_postgres_dao
from app.dto.common_dto import MessageInfo, Sender
from app.models import Mood
from app.services.llm_response import LlmResponseService, get_llm_response_service


class MoodJournalService:
    def __init__(self, postgres_dao, llm_response):
        self.__table_chat = None
        self.postgres_dao = postgres_dao
        self.llm_response = llm_response

    def get_mood_journal(self, user_id):
        return self.postgres_dao.get_mood_journal(user_id)

    def format_mood_journal_for_prompt(self, mood_journal):
        emotions = mood_journal.emotions
        # Ensure emotions is a Python list
        if isinstance(emotions, str):
            import json
            emotions = json.loads(emotions)  # Safely parse JSON string to Python list

        return f"""
    emotions: {emotions}
    description: {mood_journal.description}
    mood_score: {mood_journal.mood_score}
    anxiety_score: {mood_journal.anxiety_score}
    """

    def get_recommendation_message(self, user_id):
        mood_journal: Mood = self.get_mood_journal(user_id)
        if mood_journal:
            formatted_prompt = self.format_mood_journal_for_prompt(mood_journal)
            return self.llm_response.process_data(
                llm='gpt',
                user_input=formatted_prompt,
                system_prompt=sys_prompt_recommendation
            )
        else:
            return None

    def process_recommendation(self, user_id, chat_id, recommendation_text):
        recommendation_message = MessageInfo(
            sender=Sender.ASSISTANT,
            text=recommendation_text,
            timestamp=datetime.utcnow()
        )
        self.append_message(user_id, chat_id, recommendation_message)

    def append_message(self, user_id, chat_id, new_message: MessageInfo):
        # Update the DynamoDB table to append the message
        response = self.__table_chat.update_item(
            Key={
                'user_id': str(user_id),
                'chat_id': str(chat_id)
            },
            UpdateExpression="SET #msg = list_append(if_not_exists(#msg, :empty_list), :new_msg)",
            ExpressionAttributeNames={
                '#msg': 'messages'
            },
            ExpressionAttributeValues={
                ':empty_list': [],
                ':new_msg': [new_message.to_dict()]
            },
            ReturnValues="UPDATED_NEW"
        )
        return response


def get_mood_journal_service(postgres_dao=Depends(get_postgres_dao), llm_response=Depends(get_llm_response_service)):
    return MoodJournalService(postgres_dao, llm_response)


