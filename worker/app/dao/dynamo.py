import os
import uuid

import boto3
from decimal import Decimal

from botocore.exceptions import ClientError

from app.dto.common_dto import MessageInfo


class DynamoDbDAO:

    def __init__(self):
        self.__dynamodb = boto3.resource(
            'dynamodb',
            region_name='us-east-1'
        )
        self.__table_chat = self.__dynamodb.Table(f"chat-{os.getenv('ENV')}")
        self.__table_questionnaire_summary = self.__dynamodb.Table('questionnaire_summary')
        self.__table_chat_memory = self.__dynamodb.Table(f"chat-memory-{os.getenv('ENV')}")

    def create_initial_message(self, user_id: uuid.uuid4, chat_id: uuid.uuid4, message_info: MessageInfo):
        item = {
            'user_id': str(user_id),
            'chat_id': str(chat_id),
            'messages': [
                message_info.to_dict()
            ]
        }

        self.__table_chat.put_item(Item=item)

    def append_message(self, user_id, chat_id, new_message: MessageInfo):
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

    def get_chat_messages(self, user_id: str, chat_id: str) -> list[MessageInfo]:
        try:
            response = self.__table_chat.get_item(
                Key={
                    'chat_id': str(chat_id),
                    'user_id': str(user_id),
                }
            )

            if 'Item' in response:
                item = response['Item']
                messages_data = item.get('messages', [])

                for message in messages_data:
                    if 'q_type' in message and isinstance(message['q_type'], Decimal):
                        message['q_type'] = str(message['q_type'])

                messages = [MessageInfo(**message) for message in messages_data]
                return messages
            else:
                print(f"No messages found")
                return None
        except Exception as e:
            print(e)
            return None

    def record_exists(self, user_id, chat_id):
        try:
            response = self.__table_chat.get_item(
                Key={
                    'user_id': user_id,
                    'chat_id': chat_id
                }
            )
            return 'Item' in response  # Returns True if the item exists, False otherwise
        except ClientError as e:
            return False

    def update_questionnaire_summary(self, user_id: str, summary: str):
        """
        Updates the summary in the questionnaire_summary table for a given user_id.
        """
        try:
            response = self.__table_questionnaire_summary.update_item(
                Key={
                    'user_id': user_id
                },
                UpdateExpression="SET #summary = :new_summary",
                ExpressionAttributeNames={
                    '#summary': 'summary'
                },
                ExpressionAttributeValues={
                    ':new_summary': summary
                },
                ReturnValues="UPDATED_NEW"
            )
            return response
        except ClientError as e:
            print(f"Failed to update questionnaire summary for user_id {user_id}: {e.response['Error']['Message']}")
            raise e

    def persist_questionnaire_summary(self, user_id: str, summary: str):
        try:
            item = {
                'user_id': user_id,
                'summary': summary
            }
            self.__table_questionnaire_summary.put_item(Item=item)
            print(f"Successfully persisted questionnaire summary for user_id {user_id}")
        except ClientError as e:
            print(f"Failed to persist questionnaire summary for user_id {user_id}: {e.response['Error']['Message']}")
            raise e

    def add_chat_memory(self, user_id, chat_id, memory_text):
        item = {
            'user_id': str(user_id),
            'chat_id': str(chat_id),
            'memory': memory_text
        }
        self.__table_chat_memory.put_item(Item=item)

    def update_chat_memory(self, user_id, chat_id, memory_text):
        pass


def get_dynamodb_dao():
    return DynamoDbDAO()
