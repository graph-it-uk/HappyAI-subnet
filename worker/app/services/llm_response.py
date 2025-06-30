import json

from anthropic import Anthropic
from fastapi import Depends
from openai import OpenAI

from app.config.common_sys_prompts import sys_prompt_past_problem_classifier, sys_prompt_chat_summary
from app.dao.dynamo import DynamoDbDAO, get_dynamodb_dao
from app.dto.common_dto import MessageInfo
from app.utils.secrets import get_open_ai_client, get_claude_client


class LlmResponseService:
    def __init__(self, open_ai_client: OpenAI, claude_client: Anthropic, dynamodb_dao: DynamoDbDAO):
        self.__open_ai_client = open_ai_client
        self.__claude_client = claude_client
        self.__dynamodb_dao = dynamodb_dao

    def process_data(self, llm, user_input, temperature=0.2, model='', system_prompt=None):
        if llm == 'gpt':
            model = 'gpt-4o' if model == '' else model
            return self.__try_process(self.process_data_gpt, self.process_data_claude, user_input, model, system_prompt, temperature)
        elif llm == 'claude':
            model = 'claude-3-5-sonnet-20240620' if model == '' else model
            return self.__try_process(self.process_data_claude, self.process_data_gpt, user_input, model, system_prompt)
        else:
            raise ValueError("Invalid LLM specified. Choose 'gpt' or 'claude'.")

    def process_data_gpt_json(self, user_input, model: str = "gpt-4o", system_prompt=None) -> dict:
        chat_completion = self.__open_ai_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt},
                      {
                          "role": "user",
                          "content": user_input,
                      }],
            model=model,
            response_format={"type": "json_object"},
            temperature=0.2,
            seed=123321
        )

        res = chat_completion.choices[0].message.content
        json_output_for_page = json.loads(res)
        return json_output_for_page

    def process_data_gpt(self, user_input: str, model: str = 'gpt-4o', system_prompt=None,temperature=0.2) -> str:
        chat_completion = self.__open_ai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            model=model,
            temperature=temperature,
            seed=123321,
        )
        res = chat_completion.choices[0].message.content

        return res

    def process_data_claude(self, user_input, model='claude-3-5-sonnet-20240620', system_prompt=None, temperature=0.5):
        chat_completion = self.__claude_client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_input}
            ],
            max_tokens=4096,
            temperature=temperature
        )
        res = chat_completion.content[0].text

        return res

    def __try_process(self, primary_func, fallback_func, user_input, model, system_prompt, temperature=0.2):
        try:
            return primary_func(user_input=user_input, model=model, system_prompt=system_prompt, temperature=temperature)
        except Exception:
            return fallback_func(user_input=user_input, system_prompt=system_prompt, temperature=temperature)

    def get_past_problem(self, next_q_type, last_n_messages_str):
        if next_q_type == 2:
            past_problem = self.process_data(
                llm='gpt', user_input=last_n_messages_str, system_prompt=sys_prompt_past_problem_classifier,
                model='gpt-4o'
            )

        else:
            past_problem = False

        return past_problem

    def get_llm_response_model(self, next_q_type):
        llm_response_model = 'gpt'
        return llm_response_model

    def summarize_chat(self, user_id, chat_id):
        messages_info: list[MessageInfo] = self.__dynamodb_dao.get_chat_messages(user_id, chat_id)

        conversation_str = ''
        for message in messages_info:
            conversation_str += f'{message.sender}: {message.text}\n\n'

        summary = self.process_data('gpt',
                                    conversation_str,
                                    model='gpt-4o',
                                    system_prompt=sys_prompt_chat_summary)

        return summary


def get_llm_response_service(open_ai_client: OpenAI = Depends(get_open_ai_client),
                             claude_client: Anthropic = Depends(get_claude_client),
                             dynamodb_dao: DynamoDbDAO = Depends(get_dynamodb_dao)):
    return LlmResponseService(open_ai_client, claude_client, dynamodb_dao)
