import logging
import random
import json
from datetime import datetime

import enum
import pydantic
import requests


class Role(str, enum.Enum):
    """Message is sent by which role?"""

    user = "user"
    assistant = "assistant"
    system = "system"

class Message(pydantic.BaseModel):
    role: Role
    content: str


user_generation_prompt = """
You take part in a role-play game to help the training of a psychologists.
You are playing as 'user' with the following personality and profile:
{profile}
Be authentic, try to mimic the behaviour of the person described in the profile.
Your previous conversation with the 'assistant' is:
{messages}

Your response:
"""

dangerous_behaviour_message = 'He/she is considering dangerous behaviour such as self-harm, abuse or suicidal ideation. This information should unravel gradually during the conversation.'

class SyntheticsGenerator:
    def __init__(self, client, synth_corpus_path = "chain/synthetics/synth_corpus.json"):
        self.client = client
        self.synth_corpus_path = synth_corpus_path
        self.synth_corpus = self._get_synth_corpus()
        self.dialogs = {}

    async def generate(self):
        if len(self.dialogs) < 2:
            dialog = await self._generate_new_dialog()
        else:
            dialog = await self._continue_dialog()
        user_message = self._generate_user_response(dialog)
        dialog['messages'].append(Message(role=Role.user, content=user_message))
        self.dialogs[dialog['dialog_id']] = dialog
        return dialog

    def _generate_user_response(self, dialog):
        messages = dialog['messages']
        messages_formatted = '\n'.join([f'{m.role.upper()}: {m.content}' for m in messages])
        profile = dialog['profile']
        prompt = user_generation_prompt.format(profile=profile, messages=messages_formatted)
        user_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": 'Follow instructions precisely.'},
                          {"role": "user", "content": prompt}],
            )
        return user_response.choices[0].message.content

    def _get_synth_corpus(self):
        try:
            url = 'https://raw.githubusercontent.com/graph-it-uk/HappyAI-subnet/blob/main/app/chain/synthetics/synth_corpus.json'
            resp = requests.get(url)
            synth_corpus = json.loads(resp.text)
        except:
            logging.warning('Failed to load synthetic corpus from remote storage. Using local version instead.')
            with open(self.synth_corpus_path, "r") as fh:
                synth_corpus = json.load(fh)
        return synth_corpus

    async def _generate_new_dialog(self):
        person_profile = self._combine_person_profile()
        dialog = {'dialog_id': datetime.now().isoformat(),
                  'profile': person_profile,
                  'messages': [Message(role=Role.assistant,
                                       content="How can I help you today?")]}
        return dialog

    def _combine_person_profile(self):
        person = random.choice(self.synth_corpus['people'])
        fears = [random.choice(self.synth_corpus['fears']) for _ in range(random.randint(0, 2))]
        issues = [random.choice(self.synth_corpus['issues']) for _ in range(random.randint(0, 1))]
        traumas = [random.choice(self.synth_corpus['traumas']) for _ in range(random.randint(0, 1))]

        if len(issues) > 0:
            person += 'He/she has mental issues: ' + ', '.join(issues)
        if len(traumas) > 0:
            person += 'He/she has traumatic experiences: ' + ', '.join(traumas)
        if len(fears) > 0:
            person += 'He/she has fears: ' + ', '.join(fears)
        if random.random() < 0.1:
            person += dangerous_behaviour_message
        return person

    async def _continue_dialog(self):
        dialog_to_continue = self.dialogs[random.choice(list(self.dialogs.keys()))]
        if len(dialog_to_continue['messages']) > 20:
            self.dialogs.pop(dialog_to_continue['dialog_id'])
            dialog_to_continue = await self._generate_new_dialog()
        return dialog_to_continue

    def update_dialog(self, dialog_id, message):
        self.dialogs[dialog_id]['messages'].append(Message(role=Role.assistant, content=message))