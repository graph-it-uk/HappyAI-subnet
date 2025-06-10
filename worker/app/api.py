import re

from typing import List
from fastapi import APIRouter, Depends, HTTPException


from app.dto.common_dto import DialogRequest, DialogMessage, DialogResponse
from app.services.llm_response import get_llm_response_service
from app.services.orchestrator import get_orchestrator_service
from app.utils.secrets import get_current_user_id

router = APIRouter()


def format_messages_list_to_str(messages: list[DialogMessage]) -> str:
    return "\n\n".join(m.sender + ': ' + m.text for m in messages)


@router.post("/send_message/")
def send_message(request: DialogRequest,
                 orchestrator_service=Depends(get_orchestrator_service),
                 llm_response_service=Depends(get_llm_response_service)) -> DialogResponse:

    last_n_messages = request.dialog[-8:]
    last_messages_str = format_messages_list_to_str(request.dialog)
    last_n_messages_str = format_messages_list_to_str(last_n_messages)

    session_input: str = orchestrator_service.get_session_input(last_n_messages_str, request.user_input.text)
    next_q_type = orchestrator_service.classify_q_type(request.user_input.text, last_n_messages_str)

    is_issue = orchestrator_service.is_issue(request.user_input, last_n_messages)
    past_problem = llm_response_service.get_past_problem(next_q_type, last_n_messages_str)
    assistant_response_model = orchestrator_service.get_assistant_response_model(next_q_type)
    assistant_response_prompt = orchestrator_service.get_assistant_response_prompt(request.chat_llm_model, next_q_type,
                                                                                   past_problem, last_n_messages_str)
    llm_model = orchestrator_service.get_llm_model(session_input, next_q_type, is_issue)
    llm_response_model = llm_response_service.get_llm_response_model(next_q_type)

    #rag_service.process_chat_summary(chat, next_q_type)

    temperature = 0.7 if next_q_type['question_flag'] == 2 else 0.2

    full_session_input = orchestrator_service.get_session_input(last_messages_str, request.user_input.text)
    assistant_response: str = llm_response_service.process_data(llm_response_model,  # pass temeperature
                                                                full_session_input,
                                                                temperature,
                                                                assistant_response_model,
                                                                assistant_response_prompt)

    #chat_service.update_chat_name_and_llm_name(user_id, chat.id, chat_name, llm_model)

    return DialogResponse(request_id=request.request_id,
                          q_type = next_q_type['question_flag'],
                          assistant_message=assistant_response)