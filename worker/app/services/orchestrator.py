import json
from typing import Tuple, Any

from fastapi import Depends
from openai import OpenAI

from app.config.common_sys_prompts import sys_prompt_issue_classificator, sys_prompt_model_selector
from app.config.relationship_prompts import sys_q_type_22_relationship_past_experience, sys_q_type_22_relationship
from app.services.llm_response import get_llm_response_service
from app.config.prompts_claude import sys_prompt_orchestrator_claude, sys_q_type_2_claude_past_experience, \
    sys_q_type_2_claude
from app.config.prompts_mapper import SystemPromptsMapper
#from app.services.rag import get_rag_service
from app.utils.secrets import get_open_ai_client


class OrchestratorService:
    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.system_prompts = SystemPromptsMapper()

    def chat_name_creation(self, user_input, system_prompt):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    # "content": user_prompt,
                    "content": user_input,
                }
            ],
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            temperature=0.2,
            seed=123321,
            # stream=True
        )
        res = chat_completion.choices[0].message.content
        json_name = json.loads(res)
        return json_name['chat_session_name']

    def classify_q_type(self, user_input: str, last_n_messages: str) -> dict:
        chat_conversation = f""" Current user message: {user_input}.

        The whole session conversation: 
        {last_n_messages}
        
        """
        next_q_type_claude: dict = self.model.process_data_gpt_json(chat_conversation, system_prompt=sys_prompt_orchestrator_claude)

        return next_q_type_claude

    def is_issue(self, user_input, last_n_messages):
        chat_conversation = f""" Current user message: {user_input}.

        The whole session conversation: 
        {last_n_messages}

        """

        return self.model.process_data_gpt_json(chat_conversation, system_prompt=sys_prompt_issue_classificator)['is_issue']

    def get_session_name(self, session_input, session_name_prompt):
        return self.chat_name_creation(session_input, session_name_prompt)

    def get_current_sys_prompt(self, next_q_type: dict, model: str = 'core') -> str:
        return self.system_prompts.get_prompt(next_q_type['question_flag'], model)

    def get_session_input(self, last_n_messages, user_input):
        return last_n_messages + str('\n\nUSER: ' + user_input)

    def get_assistant_response_model(self, next_q_type):
        assistant_response_model = 'gpt-4o'
        return assistant_response_model

    def __embed_questionnaire_into_prompt(self, questions, assistant_response_prompt):
        if questions:
            assistant_response_prompt += (
                "\n\nYou will be provided with questions, the form that the user fills in. "
                "Please keep it in mind while generating your response, but never directly reference this information in your response.\n\n"
                f"{str(questions)}"
            )
        return assistant_response_prompt

    def get_assistant_response_prompt(self, chat_llm_model, next_q_type, past_problem, last_n_messages_str, questions=None):
        if  next_q_type['question_flag'] == 2:
            generated_chat_summary = last_n_messages_str #self.rag_service.query_index_with_filter(last_n_messages_str, chat.fk_user_id)
            if chat_llm_model == 'core' and past_problem:
                assistant_response_prompt = sys_q_type_2_claude_past_experience.format(
                    generated_chat_summary=generated_chat_summary)
            elif chat_llm_model == 'relationship' and past_problem:
                assistant_response_prompt = sys_q_type_22_relationship_past_experience.format(
                    generated_chat_summary=generated_chat_summary)
            elif past_problem is False and chat_llm_model == 'core':
                assistant_response_prompt = sys_q_type_2_claude
            elif past_problem is False and chat_llm_model == 'relationship':
                assistant_response_prompt = sys_q_type_22_relationship
            else:
                assistant_response_prompt = self.get_current_sys_prompt(next_q_type, chat_llm_model)

        else:
            assistant_response_prompt = self.get_current_sys_prompt(next_q_type, chat_llm_model)

        return self.__embed_questionnaire_into_prompt(questions, assistant_response_prompt)

    def get_llm_model(self, session_input, next_q_type, is_issue):
        if is_issue or next_q_type['question_flag'] == 2:
            model = self.model.process_data_gpt_json(session_input, system_prompt=sys_prompt_model_selector)[
                'selected']
        else:
            model = 'core'

        return model


def get_orchestrator_service(client: OpenAI = Depends(get_open_ai_client),
                             model = Depends(get_llm_response_service)) -> OrchestratorService:
    return OrchestratorService(client, model)
