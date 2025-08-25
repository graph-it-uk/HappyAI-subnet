from ast import literal_eval
import json
import random
import os
from dotenv import load_dotenv
import instructor

import torch

from supabase import create_client
from app.chain.evaluation.elo_system import ELOSystem
from app.chain.evaluation.models import TournamentResult
from app.chain.evaluation.prompts.PromptFacade import PromptFacade
import bittensor as bt


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

        self.worker = worker
        
        # Initialize ELO system for tournament mode
        self.elo_system = ELOSystem()

        self.prompt_facade = PromptFacade()
        
        # Initialize instructor client for structured responses
        self.instructor_client = instructor.from_openai(client)

        load_dotenv()
        self.supabase_mode = os.environ.get("SUPABASE_MODE", "False").lower() == "true"
        if self.supabase_mode:
          bt.logging.warning(f"SUPABASE_MODE: {self.supabase_mode}")
          
          self.supabase = create_client(
              supabase_url=os.environ.get("SUPABASE_URL"),
              supabase_key=os.environ.get("SUPABASE_KEY")
          )
          self.user_id = os.getpid()
          
          # Connection test removed - simplified system
          bt.logging.debug("Supabase connection established")
          

        self._initialized = True



    def evaluate(self, query, responses, miner_uids=None):
        """
        Tournament-based evaluation using weighted scoring from rubric criteria.
        
        Args:
            query: The query sent to miners
            responses: List of miner responses
            miner_uids: List of miner UIDs corresponding to responses (required for ELO)
            
        Returns:
            normalized_scores: List of scores (0.0 to 1.0) for each miner
        """
        if miner_uids is None:
            bt.logging.error("Miner UIDs required for tournament evaluation")
            return torch.zeros(len(responses)), None
            
        if len(responses) != len(miner_uids):
            bt.logging.error("Number of responses must match number of miner UIDs")
            return torch.zeros(len(responses)), None
            
        # Store miner UIDs for use in evaluation methods
        self.current_miner_uids = miner_uids
            
        bt.logging.info(f"Tournament evaluation: {len(responses)} miners")
        
        # Select 2 random criteria (at least 1 critical, different categories)
        selected_criteria = self._select_tournament_criteria(num_criteria=2)
        bt.logging.info(f"Selected criteria: {[c['id'] for c in selected_criteria]}")
        
        # Evaluate all criteria together using LLM
        tournament_result = self._evaluate_criterion_tournament(query, responses, selected_criteria)
        
        if not tournament_result or not tournament_result.miner_evaluations:
            bt.logging.error("Failed to get tournament evaluation results")
            return torch.zeros(len(responses)), None
        
        # Extract scores and apply weights
        scores = torch.zeros(len(responses))
        
        for i, (uid, response) in enumerate(zip(miner_uids, responses)):
            if not response:
                scores[i] = 0
                continue
                
            # Find evaluation for this miner
            miner_eval = None
            for eval_item in tournament_result.miner_evaluations:
                if eval_item.miner_id == uid:
                    miner_eval = eval_item
                    break
            
            if miner_eval:
                # Calculate weighted score
                weighted_scores = []
                for criterion in selected_criteria:
                    criterion_id = criterion['id']
                    weight = criterion.get('weight', 1.0)
                    
                    # Find score for this criterion
                    criterion_score = 0
                    for rubric in miner_eval.criteria_scores:
                        if rubric.rubric_id == criterion_id:
                            # Extract score from marks - the value should be the score
                            criterion_score = rubric.mark
                            break
                    
                    weighted_score = criterion_score * weight
                    weighted_scores.append(weighted_score)
                    bt.logging.debug(f"Miner {uid}: {criterion_id} = {criterion_score}/10 × {weight} = {weighted_score}")
                
                # Calculate final weighted score
                if weighted_scores:
                    scores[i] = sum(weighted_scores) / len(weighted_scores)
                    bt.logging.info(f"Miner {uid}: Final weighted score = {scores[i]:.2f}")
        
        # Convert scores to rankings for ELO update
        score_rankings = [(score.item(), idx) for idx, score in enumerate(scores)]
        score_rankings.sort(reverse=True)  # Higher scores first
        
        # Create (uid, rank) tuples for ELO update
        group_results = []
        for rank, (score, response_idx) in enumerate(score_rankings, 1):
            uid = miner_uids[response_idx]
            group_results.append((uid, rank))
            bt.logging.info(f"Miner {uid}: Rank {rank} (score: {score:.2f})")
        
        # Log tournament results (no ELO updates needed)
        bt.logging.info(f"Tournament group completed: {len(group_results)} miners evaluated")
        
        # Return raw scores for proper ELO ranking (not normalized)
        bt.logging.debug(f"Tournament evaluation completed for {len(miner_uids)} miners")
        
        return scores, None  # Return raw scores, not normalized
        
    def _select_tournament_criteria(self, num_criteria=2):
        """
        Select random criteria ensuring:
        1. At least one critical criterion
        2. Criteria from different categories
        """
        # Load judge queries for criteria selection
        judge_queries = json.load(open("app/chain/evaluation/judge_queries.jsonl"))
        
        # Separate critical and non-critical criteria
        critical_criteria = [c for c in judge_queries if c.get('critical', 0) == 1]
        non_critical_criteria = [c for c in judge_queries if c.get('critical', 0) == 0]
        
        selected = []
        used_categories = set()
        
        # Always include at least one critical criterion
        if critical_criteria:
            critical_criterion = random.choice(critical_criteria)
            selected.append(critical_criterion)
            used_categories.add(critical_criterion['category'])
            num_criteria -= 1
        
        # Fill remaining slots ensuring different categories
        remaining_criteria = critical_criteria + non_critical_criteria
        if num_criteria > 0 and remaining_criteria:
            # Filter criteria to only include unused categories
            available_criteria = [c for c in remaining_criteria if c['category'] not in used_categories]
            
            if available_criteria:
                additional = random.sample(available_criteria, min(num_criteria, len(available_criteria)))
                selected.extend(additional)
                used_categories.update([c['category'] for c in additional])
        
        bt.logging.info(f"Selected criteria: {[c['id'] for c in selected]} from categories: {list(used_categories)}")
        return selected
        
    def _evaluate_criterion_tournament(self, query, responses, criteria):
        """
        Evaluate all miner responses for selected criteria using LLM.
        Returns TournamentResult with structured evaluation.
        """
        try:
            # Prepare data for template
            template_data = {
                'criteria': criteria,
                'miners': [
                    {'uid': uid, 'response': response} 
                    for uid, response in zip(self.current_miner_uids, responses)
                ]
            }
            
            # Use PromptFacade to get the prompt
            messages = literal_eval(self.prompt_facade.get_prompt("judge", **template_data))


            
            # Call LLM with structured response
            response, completion = (
                self.instructor_client.messages.create_with_completion(
                    model="gpt-5-nano",
                    max_completion_tokens=2000,
                    messages=messages,
                    response_model=TournamentResult,
                )
            )

            print(response)
            
            bt.logging.info(f"LLM evaluation completed: {response}")
            return response
            
        except Exception as e:
            bt.logging.error(f"Failed to evaluate tournament criteria: {e}")
            # Fallback to mock evaluation if LLM fails
            return self._fallback_evaluation(responses, criteria)
    
    def _fallback_evaluation(self, responses, criteria):
        """Fallback evaluation if LLM fails."""
        bt.logging.warning("Using fallback evaluation due to LLM failure")
        
        # Create mock TournamentResult
        miner_evaluations = []
        for i, (uid, response) in enumerate(zip(self.current_miner_uids, responses)):
            if response:
                # Mock scores for each criterion
                scores = {}
                for criterion in criteria:
                    scores[criterion['id']] = random.randint(0, 10)
                
                from app.chain.evaluation.models import MinerEvaluation, Rubric
                criteria_scores = []
                for criterion_id, score in scores.items():
                    criteria_scores.append(Rubric(
                        rubric_id=criterion_id,
                        mark=score
                    ))
                
                miner_evaluations.append(MinerEvaluation(
                    miner_id=uid,
                    criteria_scores=criteria_scores
                ))
        
        return TournamentResult(miner_evaluations=miner_evaluations)
        
    def _evaluate_response_with_rubric(self, query, response, criterion):
        """
        Evaluate a single response using the rubric criteria.
        For now, using simplified scoring - will be enhanced in next step.
        """
        # Simplified evaluation for now - will use LLM in next step
        # Returns random score 0-10 for demonstration
        return random.randint(0, 10)
        

