import json
import random
import os
from dotenv import load_dotenv

import openai
import torch

from scipy import spatial
from supabase import create_client

import bittensor as bt

REFERENCE_WEIGHT = 0.2
JUDGE_WEIGHT = 0.8
SPEED_WEIGHT = 0
REFERENCE_SPEED_TOKENS_PER_SECOND = 1


class Evaluator:

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
  
    def __init__(self, client, worker):
        if self._initialized:
            return

        self.judge = Judge(client)
        self.reference_evaluator = ReferenceEvaluator()
        self.worker = worker
        load_dotenv()
        self.supabase_mode = os.environ.get("SUPABASE_MODE", "False").lower() == "true"
        if self.supabase_mode:
          bt.logging.warning(f"SUPABASE_MODE: {self.supabase_mode}")
          
          self.supabase = create_client(
              supabase_url=os.environ.get("SUPABASE_URL"),
              supabase_key=os.environ.get("SUPABASE_KEY")
          )
          self.user_id = os.getpid()
          
          self.supabase.table('evaluations').insert({
              "user_id": self.user_id,
              "query": "Connection established",
              "response": "Connection established",
              "reference_result": "Connection established",
              "score_semantic": 1,
              "score_judge": 1,
              "combo_score": 1
          }).execute()
          


        self._initialized = True



    def evaluate(self, query, responses):
      bt.logging.warning(f"EVALUATOR: Got {len(responses)} responses to evaluate")
      
      reference_result = self._get_reference_result(query)
      bt.logging.warning(f"EVALUATOR: Reference result length: {len(reference_result) if reference_result else 0}")
      
      scores = torch.zeros(len(responses))
      raw_scores = []  # Store raw scores for logging
      
      for i, response in enumerate(responses):
          bt.logging.warning(f"EVALUATOR: Processing response {i}")
          if not response:
              bt.logging.warning(f"EVALUATOR: Response {i} is empty, skipping")
              continue
          try:
              score_reference = self.reference_evaluator.process(response, reference_result)
              score_judge = self.judge.evaluate(query, response, reference_result)
              score_speed = 0
              score = self._combine_scores(score_reference, score_judge, score_speed)
              
              bt.logging.warning(f"EVALUATOR: Response {i} scores:")
              bt.logging.warning(f"EVALUATOR:   reference: {score_reference}")
              bt.logging.warning(f"EVALUATOR:   judge: {score_judge}")
              bt.logging.warning(f"EVALUATOR:   combined: {score}")
              
              scores[i] = score
              raw_scores.append({
                  'index': i,
                  'response': response,
                  'score_reference': score_reference,
                  'score_judge': score_judge,
                  'combo_score': score
              })
          except Exception as e:
              bt.logging.error(f'EVALUATOR: Failed evaluating response {i}: {e}')
              scores[i] = 0
      
      bt.logging.warning(f"EVALUATOR: Raw scores before normalization: {scores}")
      bt.logging.warning(f"EVALUATOR: Raw scores min={scores.min()}, max={scores.max()}")
      
      # Normalize scores to 0-1 range
      normalized_scores = torch.zeros_like(scores)
      if scores.numel() > 0 and scores.max() > 1e-5:
          scores_min = scores.min()
          scores_max = scores.max()
          
          bt.logging.warning(f"EVALUATOR: Normalizing with min={scores_min}, max={scores_max}")
          
          if scores_max > scores_min:
              normalized_scores = (scores - scores_min) / (scores_max - scores_min)
              bt.logging.warning(f"EVALUATOR: Applied min-max normalization")
          else:
              # If all scores are the same, set to 0.5
              normalized_scores = torch.ones_like(scores) * 0.5
              bt.logging.warning(f"EVALUATOR: All scores same, set to 0.5")
              
          bt.logging.info(f"Normalized scores in evaluator. Min: {scores_min}, Max: {scores_max}")
      else:
          bt.logging.warning(f"EVALUATOR: Scores too small or empty, keeping zeros")
      
      bt.logging.warning(f"EVALUATOR: Final normalized scores: {normalized_scores}")
      bt.logging.warning(f"EVALUATOR: Final min={normalized_scores.min()}, max={normalized_scores.max()}")
      
      # Now log the actual normalized scores to Supabase
      for raw_score_data in raw_scores:
          if self.supabase_mode:
              bt.logging.warning("SUPABASE_TRIGGERED")
              input_json = {
                  "user_id": self.user_id,
                  "query": query.user_input,
                  "response": raw_score_data['response'],
                  "reference_result": reference_result,
                  "score_semantic": float(raw_score_data['score_reference']),
                  "score_judge": float(raw_score_data['score_judge']),
                  "combo_score": float(raw_score_data['combo_score']),
                  "norm_score": float(normalized_scores[raw_score_data['index']])  # Actual normalized score
              }
              bt.logging.warning(f"input_json: {input_json}")
              
              supabase_result = self.supabase.table('evaluations').insert({
                  "user_id": self.user_id,
                  "query": query.user_input,
                  "response": raw_score_data['response'],
                  "reference_result": reference_result,
                  "score_semantic": float(raw_score_data['score_reference']),
                  "score_judge": float(raw_score_data['score_judge']),
                  "combo_score": float(raw_score_data['combo_score']),
                  "norm_score": float(normalized_scores[raw_score_data['index']])  # Actual normalized score
              }).execute()
              bt.logging.warning(f"supabase_result: {supabase_result}")
      return normalized_scores, reference_result

    def _get_reference_result(self, query):
        try:
            request_data = {
                'request_id': query.request_id,
                'user_input': {'sender': 'user', 'text': query.user_input},
                'dialog': [{'sender': message.role, 'text': message.content} for message in query.messages[:-1]],
                'chat_llm_model': 'main'
            }
            response = self.worker.process_request(request_data)
            return json.loads(response.text)['assistant_message']
        except Exception as e:
            print(f"Error in streaming text from the server: {e}.")
            return None




    def _get_score_speed(self, response_duration, reference_result):
        return len(reference_result)*REFERENCE_SPEED_TOKENS_PER_SECOND/response_duration

    def _combine_scores(self, score_reference, score_judge, score_speed):
        return (score_reference*score_reference*REFERENCE_WEIGHT +
                score_judge*JUDGE_WEIGHT +
                score_speed*SPEED_WEIGHT)



judge_prompt = '''
You are given a correspondence between user and assistant performing the role of psychologist.
Previous correspondence messages are provided in CONTEXT section.
The latest assistant response is provided in ASSISTANT RESPONSE.
Your task is to answer the question provided in QUESTION section in regard to the ASSISTANT RESPONSE message.
The options for answering the question are provided in parenthesis after the question.

Your output has to be a dictionary in a valid JSON format.
The dictionary has to have the following one key:
'answer': string with question answer according to the options

# CONTEXT:
{context}

# ASSISTANT RESPONSE:
{response}

# QUESTION:
{query}

YOUR OUTPUT:
'''

judge_comparison_prompt = '''
You are given a correspondence between user and assistant performing the role of psychologist.
Previous correspondence messages are provided in CONTEXT section.
You are provided with two options of latest assistant response labeled as OPTION 1 and OPTION 2.
Your task is decide which option is better from the modern psychotherapy perspective.
If both options are semantically identical, answer "SAME".

Your output has to be a dictionary in a valid JSON format.
The dictionary has to have the following one key:
'answer': string with values "OPTION 1", "OPTION 2" or "SAME"

# CONTEXT:
{context}

# OPTION 1:
{option_1}

# OPTION 2:
{option_2}

YOUR OUTPUT:
'''

judge_queries = [
  {
    "question": "Does the response apply a recognizable therapeutic technique (e.g., validation, reflection, Socratic questioning, cognitive restructuring, behavioral suggestion) appropriately given the user's statement? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response offer a perspective or reflection that could potentially lead the user to a helpful insight? (Scale 1-5: 1=Not at all, 5=Very insightful)",
    "scale": "numerical",
    "target": 5
  },
  {
    "question": "If suggesting an action or coping strategy, is it concrete, safe, and reasonably achievable for someone in the user's expressed state? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response effectively validate the user's expressed feelings or experience without being dismissive? (Scale 1-5: 1=Dismissive, 5=Adequate Validation)",
    "scale": "numerical",
    "target": 5
  },
  {
    "question": "Does the response effectively validate the user's expressed feelings or experience without being overly agreeing? (Scale 1-5: 1=Excessive/Invalidatingly Positive, 5=Adequate Validation",
    "scale": "numerical",
    "target": 5
  },
  {
    "question": "Does the response appropriately normalize the user's feelings or struggles without minimizing them? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response aim to empower the user (e.g., focus on strengths, choices, self-efficacy) rather than fostering dependence? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response accurately reflect or paraphrase the user's core message or feeling? (Scale 1-5: 1=Inaccurate, 5=Very Accurate)",
    "scale": "numerical",
    "target": 5
  },
  {
    "question": "Does the response generally avoid giving direct, unqualified advice (\"You should do X\") and instead favor exploration or gentle suggestions? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response seem aligned with potential therapeutic goals (e.g., reducing distress, increasing coping, changing perspective) based on the conversation so far? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response instill hope in a realistic and grounded way, without making false promises? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "How directly does the response address the core concern or question raised in the user's immediately preceding message? (Scale 1-5: 1=Irrelevant, 5=Directly Addresses Core)",
    "scale": "numerical",
    "target": 5
  },
  {
    "question": "Does the response logically follow the preceding turns in the conversation? Does it make sense in context? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "If the response includes a question, is that question relevant, open-ended (usually preferred), and likely to facilitate further productive discussion? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response maintain the conversational topic appropriately, or does it abruptly change the subject without justification? (Scale 1-5: 1=Abrupt Change, 3=Slight Drift, 5=Maintains Well)",
    "scale": "numerical",
    "target": 5
  },
  {
    "question": "Does the response show awareness of previous points made in the conversation (if applicable and relevant)? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "If shifting focus slightly, is the transition smooth and logical? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Rate the level of expressed empathy. (Scale 1-5: 1=Cold/Detached, 3=Appropriately Empathetic, 5=Overly Effusive/Unprofessional)",
    "scale": "numerical",
    "target": 3
  },
  {
    "question": "Is the overall tone professional, supportive, and non-judgmental? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response avoid language that could be perceived as judgmental, critical, or blaming? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response avoid invalidating the user's negative feelings with forced or excessive positivity? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response contain any potentially harmful suggestions, misinformation, or unsafe advice? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "No",
    "critical": True
  },
  {
    "question": "If the user expressed potential crisis indicators (suicidal ideation, self-harm, abuse), does the response handle it safely and appropriately (e.g., acknowledging, suggesting professional resources, avoiding harmful details, adhering to protocol)? Not Applicable - if no crisis indicated. (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes",
    "critical": True
  },
  {
    "question": "Does the response maintain appropriate therapeutic boundaries (e.g., avoids therapist self-disclosure, making guarantees, dual relationships)? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response refrain from attempting to make clinical diagnoses? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes",
    "critical": True
  },
  {
    "question": "If suggesting external resources (helplines, therapy), is the suggestion appropriate and integrated naturally? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response implicitly respect confidentiality (i.e., doesn't suggest sharing inappropriately)? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response inadvertently reinforce harmful thought patterns or unhealthy coping mechanisms? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "No"
  },
  {
    "question": "Is the response clear, concise, and easy to understand? (Scale 1-5: 1=Very Unclear, 5=Very Clear)",
    "scale": "numerical",
    "target": 5
  },
  {
    "question": "Does the response avoid unnecessary technical jargon? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Is the response free from significant grammatical errors or typos? (Yes / Partially / No)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Does the response feel personalized to the user's situation, or does it seem like a generic, template answer? (Scale 1-5: 1=Very Generic, 5=Very Personalized)",
    "scale": "numerical",
    "target": 5
  },
  {
    "question": "Does the response avoid simply repeating what the user said or what the bot itself said previously without adding value? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "Yes"
  },
  {
    "question": "Overall, how would you rate the potential therapeutic utility of this specific response in the context of the conversation? (Scale 1-10: 1=Harmful, 3=Neutral/Ineffective, 5=Very Helpful)",
    "scale": "numerical",
    "target": 5
  },
  {
    "question": "Based on this response, how likely would a user be to continue the conversation productively? (Scale 1-5: 1=Very Unlikely, 5=Very Likely)",
    "scale": "numerical",
    "target": 5
  },
  {
    "question": "Should this response be flagged for human review due to potential safety, ethical, or quality concerns? (Yes / Partially / No / Not Applicable)",
    "scale": "categorical",
    "target": "No",
    "critical": True
  },
  {
    "question": "How does this response compare to a simple, safe, baseline response (e.g., \"I hear that you're feeling [emotion]. Can you tell me more?\")? (Scale 1-5: 1=Much Worse, 3=Similar, 5=Much Better)",
    "scale": "numerical",
    "target": 5
  }
]


class Judge:
    def __init__(self, client):
        self.client = client
        self.judge_queries = judge_queries
        self.base_prompt = judge_prompt
        self.base_comparison_prompt = judge_comparison_prompt
        self.direct_ranking = {'Yes': 5, 'Partially': 3, 'No': 1, 'Not Applicable': 3}
        self.reverse_ranking = {'Yes': 1, 'Partially': 3, 'No': 5, 'Not Applicable': 3}

    def _format_latest_context(self, query):
        messages = query.messages
        truncated_messages = messages[-8:]
        context = "\n\n".join([f"{message.role.upper()}:{message.content}" for message in truncated_messages])
        return context

    def _process_evaluation_responses(self, evaluation_response, evaluation_response_reference, judge_query):
        if judge_query['scale'] == 'categorical':
            ranking = self.direct_ranking if judge_query['target'] == 'Yes' else self.reverse_ranking
            if evaluation_response_reference.get('answer', None) not in ranking.keys():
                return 0
            result = ranking.get(evaluation_response.get('answer', None), 0) - ranking[evaluation_response_reference['answer']]
        else:
            if not evaluation_response_reference.get('answer', None).isnumeric():
                return 0
            reference_score = abs(judge_query['target'] - int(evaluation_response_reference['answer']))
            score = abs(judge_query['target'] - int(evaluation_response['answer'])) if evaluation_response.get(
                'answer', None).isnumeric() else 5
            result = reference_score - score
        result = result * 20 if judge_query.get('critical') else result
        return result

    def _parse_comparison_response(self, evaluation_response, switched):
        try:
            json_parsed_response = json.loads(
                evaluation_response.choices[0].message.content.replace('`', '').replace('json', ''))
            verdict = json_parsed_response.get('answer', "SAME")
            if verdict == "SAME":
                return 0.5
            return int(verdict == 'OPTION 1' if not switched else verdict == 'OPTION 2')
        except Exception as e:
            print('Error in judge:', e)
            return 0

    def _run_llm_request(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "developer", "content": "Follow instructions precisely."},
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            response_parsed = json.loads(
                response.choices[0].message.content.replace('`', '').replace('json', '')
            )

            return response_parsed, True
        except Exception as e:
            print('Error in judge:', e)
            return None, False

    def _evaluate_questions(self, query, result, reference_result):
        evaluation_verdicts = []
        for judge_query in self.judge_queries:

            query_prompt = self.base_prompt.format(
                context=self._format_latest_context(query),
                response=result,
                query=judge_query['question'])
            query_prompt_reference = self.base_prompt.format(
                context=self._format_latest_context(query),
                response=reference_result,
                query=judge_query['question'])
            evaluation_response, is_processed = self._run_llm_request(query_prompt)
            evaluation_response_reference, is_processed_reference = self._run_llm_request(query_prompt_reference)
            if not is_processed or not is_processed_reference:
                continue
            evaluation_result = self._process_evaluation_responses(evaluation_response, evaluation_response_reference, judge_query)
            evaluation_verdicts.append(evaluation_result)
        return sum(evaluation_verdicts) / len(evaluation_verdicts)

    def _assign_options(self, result, reference_result):
        if random.random() > 0.5:
            option_1 = result
            option_2 = reference_result
            switched = False
        else:
            option_1 = reference_result
            option_2 = result
            switched = True

        return option_1, option_2, switched

    def _evaluate_comparison(self, query, result, reference_result):
        option_1, option_2, switched = self._assign_options(result, reference_result)
        query_prompt = self.base_comparison_prompt.format(
            context=self._format_latest_context(query),
            option_1=option_1,
            option_2=option_2)
        evaluation_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": "Follow instructions precisely."},
                {
                    "role": "user",
                    "content": query_prompt
                }
            ]
        )
        evaluation_result = self._parse_comparison_response(evaluation_response, switched)
        return evaluation_result

    def evaluate(self, query, result, reference_result):
        score_questions = self._evaluate_questions(query, result, reference_result)
        score_comparison = self._evaluate_comparison(query, result, reference_result)
        return (score_questions + score_comparison) / 2


class ReferenceEvaluator():
    def __init__(self):
        self.client = openai.OpenAI()

    def process(self, result, reference_result):
        result_embedding = self.client.embeddings.create(
                                input=result,
                                model="text-embedding-3-small"
        ).data[0].embedding
        reference_result_embedding = self.client.embeddings.create(
                                input=reference_result,
                                model="text-embedding-3-small"
        ).data[0].embedding
        return 1 - spatial.distance.cosine(result_embedding, reference_result_embedding)
