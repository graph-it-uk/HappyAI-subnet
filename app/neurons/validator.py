import os
import time
import json
from datetime import datetime
from traceback import print_exception
from typing import List, Dict

import bittensor as bt
import openai
from dotenv import load_dotenv
import asyncio
import torch

from app.chain.base.validator import BaseValidatorNeuron
from app.chain.evaluation.evaluator import Evaluator
from app.chain.evaluation.simple_elo_sync import EloManager
from app.chain.protocol import CompletionSynapse
from app.chain.synthetics.generator import SyntheticsGenerator

from app.chain.utils.uids import tournament_group_miners
from app.chain.worker import Worker
from supabase import create_client


class Validator(BaseValidatorNeuron):
    def __init__(self):
        super(Validator, self).__init__()
        load_dotenv()

        # for ranking results evaluation
        llm_client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
            max_retries=3,
        )
        self.llm_client = llm_client
        self.synthetics_generator = SyntheticsGenerator(llm_client)
        self.worker = Worker(worker_url=os.environ["WORKER_URL"], worker_port=os.environ["WORKER_PORT"])
        
        # Initialize Supabase configuration FIRST
        self.supabase_mode = os.environ.get("SUPABASE_MODE", "False").lower() == "true"
        self.supabase = None
        if self.supabase_mode:
            try:
                supabase_url = os.environ.get("SUPABASE_URL")
                supabase_key = os.environ.get("SUPABASE_KEY")
                if supabase_url and supabase_key:
                    self.supabase = create_client(supabase_url, supabase_key)
                    bt.logging.info("✅ Supabase client initialized successfully")
                else:
                    bt.logging.warning("⚠️ Supabase credentials not found, Supabase disabled")
            except Exception as e:
                bt.logging.error(f"❌ Failed to initialize Supabase client: {e}")
                self.supabase = None
        
        # Initialize ELO sync manager if Supabase is enabled
        self.elo_manager = None
        if self.supabase_mode and self.supabase:
            try:
                supabase_url = os.environ.get("SUPABASE_URL")
                supabase_key = os.environ.get("SUPABASE_KEY")
                if supabase_url and supabase_key:
                    self.elo_manager = EloManager(supabase_url, supabase_key, self)
                    bt.logging.info("ELO sync initialized successfully with validator instance")
                else:
                    bt.logging.warning("Supabase credentials not found, ELO sync disabled")
            except Exception as e:
                bt.logging.error(f"Failed to initialize ELO sync manager: {e}")
                self.elo_manager = None
        
        # Initialize evaluator
        self.evaluator = Evaluator(llm_client, self.worker)
        
        # Epoch management
        self.current_epoch = 0
        self.epoch_duration = 600  # 10 minutes per epoch
        self.tournament_group_size = 6  # Default group size for tournaments
        self.num_evaluation_rounds = 3  
        self.evaluation_round = 0
        
        # Store previous ELO scores for miners
        self.previous_elo_scores = {}
        
        self.banned_coldkeys = set()
        self.load_banned_miners()
        
        # Set validator UID for ELO sync (will be updated in sync)
        self.validator_uid = None


    def load_banned_miners(self):
        """Load banned miners from banned_miners.json file"""
        try:
            with open('banned_miners.json', 'r') as f:
                ban_config = json.load(f)
                self.banned_coldkeys = set(ban_config.get('banned_coldkeys', []))
                bt.logging.info(f"Loaded {len(self.banned_coldkeys)} banned coldkeys from banned_miners.json")
                if self.banned_coldkeys:
                    bt.logging.warning(f"Banned coldkeys: {list(self.banned_coldkeys)}")
        except FileNotFoundError:
            bt.logging.info("No banned_miners.json found, starting with empty ban list")
            self.banned_coldkeys = set()
        except Exception as e:
            bt.logging.error(f"Error loading banned_miners.json: {e}")
            self.banned_coldkeys = set()

    def get_banned_miner_uids(self):
        """Get UIDs of currently banned miners"""
        banned_uids = []
        
        if not self.banned_coldkeys:
            return banned_uids
            
        for uid in range(self.metagraph.n.item()):
            try:
                # Get neuron info to access coldkey
                neuron_info = self.subtensor.neuron_for_uid(uid, self.config.netuid)
                if neuron_info and neuron_info.coldkey in self.banned_coldkeys:
                    banned_uids.append(uid)
                    bt.logging.debug(f"Found banned miner: UID {uid} with coldkey {neuron_info.coldkey}")
            except Exception as e:
                bt.logging.trace(f"Could not get coldkey for UID {uid}: {e}")
                continue
                
        if banned_uids:
            bt.logging.warning(f"Excluded {len(banned_uids)} banned miners from validation: UIDs {banned_uids}")
        
        return banned_uids

    def reload_banned_miners(self):
        """Reload banned miners from file - useful for updating bans without restart"""
        bt.logging.info("Reloading banned miners list...")
        self.load_banned_miners()
    
    def get_previous_elo_scores(self, miner_uids: List[int]) -> Dict[int, float]:
        """
        Get previous ELO scores for miners from this validator's final scores.
        Accumulates scores across epochs without normalization.
        
        Args:
            miner_uids: List of miner UIDs to get scores for
            
        Returns:
            Dict mapping miner UID to accumulated ELO score
        """
        if not self.elo_manager:
            bt.logging.debug("No ELO sync manager available")
            return {}
            
        try:
            previous_scores = {}
            
            # Get validator hotkey
            if not hasattr(self, 'validator_uid') or self.validator_uid is None:
                bt.logging.debug("Validator UID not set yet")
                return {}
                
            if self.validator_uid >= len(self.metagraph.hotkeys):
                bt.logging.warning("Validator UID out of bounds")
                return {}
                
            validator_hotkey = self.metagraph.hotkeys[self.validator_uid]
            
            for uid in miner_uids:
                # Get miner hotkey from metagraph
                if uid >= len(self.metagraph.hotkeys):
                    bt.logging.warning(f"Miner UID {uid} out of bounds")
                    previous_scores[uid] = 1000  # Default ELO rating
                    continue
                
                miner_hotkey = self.metagraph.hotkeys[uid]
                
                # Get the most recent final ELO score for this miner from this validator
                result = self.supabase.table('validator_final_scores').select('final_elo').eq('validator_hotkey', validator_hotkey).eq('miner_hotkey', miner_hotkey).order('epoch', desc=True).limit(1).execute()
                
                if result.data:
                    # Get the final ELO score (not normalized)
                    final_elo = result.data[0]['final_elo']
                    previous_scores[uid] = final_elo
                    bt.logging.debug(f"Miner {uid} ({miner_hotkey}): Previous ELO score {final_elo}")
                else:
                    # No previous score, use default
                    previous_scores[uid] = 1000  # Default ELO rating
                    bt.logging.debug(f"Miner {uid} ({miner_hotkey}): No previous score, using default 1000")
            
            bt.logging.info(f"Retrieved previous ELO scores for {len(previous_scores)} miners")
            return previous_scores
            
        except Exception as e:
            bt.logging.error(f"Error getting previous ELO scores: {e}")
            return {}
    
    def store_current_elo_scores(self, miner_uids: List[int], scores: List[float]):
        """
        Store current ELO scores for the next validation cycle.
        
        Args:
            miner_uids: List of miner UIDs
            scores: List of corresponding scores
        """
        try:
            for uid, score in zip(miner_uids, scores):
                self.previous_elo_scores[uid] = score
                bt.logging.debug(f"Stored current score for miner {uid}: {score:.4f}")
            
            bt.logging.info(f"Stored current ELO scores for {len(miner_uids)} miners")
            
        except Exception as e:
            bt.logging.error(f"Error storing current ELO scores: {e}")
    
    def refresh_elo_scores(self):
        """
        Refresh ELO scores every 10 epochs.
        This resets the previous scores to default (1000) to force fresh evaluation.
        """
        try:
            bt.logging.info("🔄 Refreshing ELO scores every 10 epochs")
            self.previous_elo_scores.clear()
            bt.logging.info("✅ ELO scores refreshed to default (1000), starting fresh evaluation")
            
        except Exception as e:
            bt.logging.error(f"Error refreshing ELO scores: {e}")
    

    
    def submit_elo_ratings(self, epoch: int, miner_uids: List[int], rewards: List[float]):
        """
        Submit ELO ratings for evaluated miners.
        Accumulates new scores with previous scores.
        
        Args:
            epoch: Current epoch number
            miner_uids: List of miner UIDs evaluated
            rewards: List of reward scores for each miner
        """
        if not self.elo_manager:
            bt.logging.debug("No ELO sync manager available")
            return
            
        if not self.validator_uid:
            bt.logging.warning("Validator UID not set, cannot submit ratings")
            return
            
        try:
            # Get previous ELO scores for accumulation
            previous_scores = self.get_previous_elo_scores(miner_uids)
            
            # Submit accumulated ELO ratings for evaluated miners
            for uid, reward in zip(miner_uids, rewards):
                if reward > 0:  # Only submit positive ratings
                    # Get previous score or use default
                    previous_elo = previous_scores.get(uid, 1000)
                    
                    # Add ELO bonus to previous score (evaluator now gives ELO bonuses directly)
                    elo_bonus = int(reward)  # reward is now the ELO bonus (36, 30, 24, etc.)
                    accumulated_elo = previous_elo + elo_bonus
                    
                    # Get miner hotkey from metagraph
                    if uid >= len(self.metagraph.hotkeys):
                        bt.logging.warning(f"Miner UID {uid} out of bounds, skipping")
                        continue
                    
                    miner_hotkey = self.metagraph.hotkeys[uid]
                    validator_hotkey = self.metagraph.hotkeys[self.validator_uid] if self.validator_uid < len(self.metagraph.hotkeys) else None
                    
                    if validator_hotkey:
                        success = self.elo_manager.submit_elo_rating(self.current_epoch, miner_hotkey, accumulated_elo, validator_hotkey)
                        if success:
                            bt.logging.debug(f"✅ ELO rating {accumulated_elo} (previous: {previous_elo} + bonus: +{elo_bonus}) submitted for miner {uid} ({miner_hotkey})")
                        else:
                            bt.logging.warning(f"⚠️ Failed to submit ELO rating for miner {uid} ({miner_hotkey})")
                    else:
                        bt.logging.error(f"❌ Validator hotkey not available for UID {self.validator_uid}")
                        
        except Exception as e:
            bt.logging.error(f"Error submitting ELO ratings: {e}")

    def calculate_and_set_elo_weights(self, final_scores: Dict[str, Dict]):
        """
        Calculate normalized weights (0 to 1) based on ELO scores and push to blockchain.
        
        Args:
            final_scores: Dict mapping miner_hotkey to {final_elo, uid}
        """
        try:
            if not final_scores:
                bt.logging.warning("No final scores available for weight calculation")
                return
            
            # Extract ELO scores and UIDs
            elo_data = []
            for miner_hotkey, score_data in final_scores.items():
                uid = score_data['uid']
                final_elo = score_data['final_elo']
                elo_data.append((uid, final_elo))
            
            if not elo_data:
                bt.logging.warning("No valid ELO data for weight calculation")
                return
            
            # Sort by ELO score (lowest to highest)
            elo_data.sort(key=lambda x: x[1])
            
            # Create weight tensor
            weights = torch.zeros(self.metagraph.n.item())
            
            # Calculate normalized weights (0 to 1)
            min_elo = elo_data[0][1]  # Lowest ELO score
            max_elo = elo_data[-1][1]  # Highest ELO score
            
            if max_elo == min_elo:
                # All miners have same ELO - give equal weights
                for uid, _ in elo_data:
                    weights[uid] = 1.0 / len(elo_data)
                bt.logging.info(f"All miners have same ELO ({min_elo}), using equal weights")
            else:
                # Normalize from 0 to 1
                for uid, elo_score in elo_data:
                    # Normalize: (elo - min_elo) / (max_elo - min_elo)
                    normalized_weight = (elo_score - min_elo) / (max_elo - min_elo)
                    weights[uid] = normalized_weight
                    bt.logging.info(f"Miner {uid}: ELO {elo_score} -> Weight {normalized_weight:.4f}")
                
                bt.logging.info(f"ELO normalization: min={min_elo}, max={max_elo}")
            
            # Store weights for use in set_weights method
            self.elo_weights = weights
            
            # Push weights to blockchain immediately
            self.set_weights()
            
            bt.logging.info(f"✅ ELO weights calculated and pushed to blockchain: {len(elo_data)} miners")
            
        except Exception as e:
            bt.logging.error(f"Error calculating ELO weights: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))

    def should_set_weights(self) -> bool:
        """
        Override base neuron logic to ensure weights are set every epoch.
        This is critical for the tournament evaluation system.
        """
        # Always set weights after tournament evaluation
        if hasattr(self, 'elo_weights') and self.elo_weights is not None:
            return True
        
        # Fallback to base logic for other cases
        return super().should_set_weights()

    async def forward(self):
        """
        Validator forward pass with tournament evaluation.
        """
        try:
            # Get banned miner UIDs to exclude from selection
            banned_uids = self.get_banned_miner_uids()
            
            # Get all tournament groups for this epoch
            all_groups = tournament_group_miners(self, banned_uids, None, self.tournament_group_size)
            bt.logging.info(f"Created {len(all_groups)} tournament groups for epoch {self.current_epoch}")
            
            # Get all unique miner UIDs from all groups
            all_miner_uids = list(set(uid for group in all_groups for uid in group))
            
            # Load previous ELO scores for all miners
            previous_scores = self.get_previous_elo_scores(all_miner_uids)
            bt.logging.info(f"Loaded previous ELO scores for {len(previous_scores)} miners")
            
            # Debug: Log first few groups to see the structure
            if all_groups:
                bt.logging.debug(f"First 3 groups: {all_groups[:3]}")
                bt.logging.debug(f"Total miners in groups: {sum(len(group) for group in all_groups)}")
                bt.logging.debug(f"Unique miners: {len(set(uid for group in all_groups for uid in group))}")
            else:
                bt.logging.warning("No tournament groups created - this will cause evaluation to fail")
            
            if not all_groups:
                bt.logging.warning("No tournament groups created, skipping evaluation")
                return
            
            # Generate synthetic query for all groups
            synthetic_dialog = await self.synthetics_generator.generate()
            query = CompletionSynapse(request_id = int(datetime.now().timestamp()*1000),
                                    messages = synthetic_dialog['messages'][:-1],
                                    user_input = synthetic_dialog['messages'][-1].content)
            bt.logging.info('Generated synthetic query for tournament evaluation')
            
            # Local score collection - store scores for each miner across all groups
            miner_scores = {}  # uid -> list of scores
            total_miners_evaluated = 0
            
            # Process each tournament group and collect scores locally
            for group_idx, miner_uids in enumerate(all_groups):
                bt.logging.info(f"Evaluating tournament group {group_idx + 1}/{len(all_groups)}: {miner_uids}")
                
                # Query miners in this group with increased timeout and retry logic
                max_retries = 2
                responses = None
                
                for attempt in range(max_retries + 1):
                    try:
                        bt.logging.debug(f"Querying group {group_idx + 1} miners (attempt {attempt + 1}/{max_retries + 1})")
                        responses = await self.dendrite(
                            axons=[self.metagraph.axons[uid] for uid in miner_uids],
                            synapse=query,
                            deserialize=True,
                            timeout=45,  # Increased timeout from 20 to 45 seconds
                        )
                        bt.logging.debug(f"Group {group_idx + 1}: Successfully received responses on attempt {attempt + 1}")
                        break
                    except Exception as e:
                        if attempt < max_retries:
                            bt.logging.warning(f"Group {group_idx + 1}: Attempt {attempt + 1} failed with {type(e).__name__}: {e}. Retrying...")
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                        else:
                            bt.logging.error(f"Group {group_idx + 1}: All {max_retries + 1} attempts failed. Last error: {type(e).__name__}: {e}")
                            responses = [None] * len(miner_uids)  # Create empty responses for failed miners
                
                if responses is None:
                    bt.logging.error(f"Group {group_idx + 1}: Failed to get any responses after all retries")
                    responses = [None] * len(miner_uids)
                
                # Filter working responses with relaxed validation
                working_miner_indices = []
                working_responses = []
                working_uids = []
                
                for idx, (uid, response) in enumerate(zip(miner_uids, responses)):
                    # Debug logging for response structure
                    bt.logging.debug(f"Miner {uid} response type: {type(response)}")
                    if response:
                        bt.logging.debug(f"Miner {uid} response attributes: {dir(response)}")
                        if hasattr(response, 'results'):
                            bt.logging.debug(f"Miner {uid} results: {response.results}")
                        if hasattr(response, 'response'):
                            bt.logging.debug(f"Miner {uid} response.response: {response.response}")
                    
                    # More flexible response validation - check multiple possible response formats
                    has_response = False
                    response_content = None
                    
                    if response:
                        # Check for results attribute (correct format from miner)
                        if hasattr(response, 'results') and response.results:
                            has_response = True
                            response_content = response.results
                            bt.logging.debug(f"Miner {uid}: Valid response in 'results' attribute")
                        # Check for response attribute (fallback)
                        elif hasattr(response, 'response') and response.response:
                            has_response = True
                            response_content = response.response
                            bt.logging.debug(f"Miner {uid}: Valid response in 'response' attribute")
                        # Check if response itself is a string (direct response)
                        elif isinstance(response, str) and response.strip():
                            has_response = True
                            response_content = response
                            bt.logging.debug(f"Miner {uid}: Valid direct string response")
                        # Check if response has any content (last resort)
                        elif response and str(response).strip():
                            has_response = True
                            response_content = str(response)
                            bt.logging.debug(f"Miner {uid}: Valid response content found")
                        else:
                            bt.logging.debug(f"Miner {uid}: No valid response content found")
                    else:
                        bt.logging.debug(f"Miner {uid}: No response object")
                    
                    if has_response and response_content:
                        working_miner_indices.append(idx)
                        working_responses.append(response_content)
                        working_uids.append(uid)
                        bt.logging.info(f"Miner {uid}: Added to working miners with content length: {len(str(response_content))}")
                    else:
                        bt.logging.warning(f"Miner {uid}: Response validation failed - no usable content")
                
                bt.logging.info(f"Group {group_idx + 1}: {len(working_responses)}/{len(miner_uids)} working miners")
                
                # Additional debug info for troubleshooting
                if len(working_responses) == 0:
                    bt.logging.warning(f"Group {group_idx + 1}: No working miners found. Response details:")
                    for i, (uid, response) in enumerate(zip(miner_uids, responses)):
                        if response is None:
                            bt.logging.warning(f"  Miner {uid}: Response is None")
                        else:
                            bt.logging.warning(f"  Miner {uid}: Response type={type(response)}, has_results={hasattr(response, 'results')}, has_response={hasattr(response, 'response')}")
                            if hasattr(response, 'results'):
                                bt.logging.warning(f"    results content: {response.results}")
                            if hasattr(response, 'response'):
                                bt.logging.warning(f"    response.response content: {response.response}")
                else:
                    bt.logging.info(f"Group {group_idx + 1}: Working miners: {working_uids}")
                    for uid, content in zip(working_uids, working_responses):
                        bt.logging.debug(f"  Miner {uid}: Content length {len(str(content))}")
                
                # Evaluate working miners in tournament
                if working_responses and len(working_responses) >= 2:  # Need at least 2 miners for tournament
                    try:
                        # Update evaluation round for this group
                        self.evaluation_round = group_idx
                        
                        # Run multiple evaluation rounds for this group
                        group_scores = {}  # uid -> list of scores from all rounds
                        
                        for round_num in range(self.num_evaluation_rounds):
                            bt.logging.info(f"Group {group_idx + 1}: Running evaluation round {round_num + 1}/{self.num_evaluation_rounds}")
                            
                            try:
                                # Evaluate using tournament system (this runs multiple criteria internally)
                                working_rewards, _ = self.evaluator.evaluate(query, working_responses, working_uids)
                                
                                # Ensure working_rewards is a valid tensor/list
                                if working_rewards is None or len(working_rewards) != len(working_uids):
                                    bt.logging.error(f"Group {group_idx + 1}, Round {round_num + 1}: Invalid evaluation results")
                                    continue
                                
                                bt.logging.info(f"Group {group_idx + 1}, Round {round_num + 1}: Results {working_rewards}")
                                
                                # Store scores from this round
                                for uid, reward in zip(working_uids, working_rewards):
                                    if uid not in group_scores:
                                        group_scores[uid] = []
                                    group_scores[uid].append(reward)
                                    bt.logging.debug(f"Miner {uid}: Round {round_num + 1} score: {reward:.4f}")
                                    
                            except Exception as e:
                                bt.logging.error(f"Group {group_idx + 1}, Round {round_num + 1} failed: {e}")
                                continue
                        
                        # Calculate average score for each miner in this group
                        for uid, round_scores in group_scores.items():
                            if uid not in miner_scores:
                                miner_scores[uid] = []
                            
                            # Calculate average score for this miner across all rounds
                            avg_group_score = sum(round_scores) / len(round_scores)
                            miner_scores[uid].append(avg_group_score)
                            
                            bt.logging.info(f"Miner {uid}: Group {group_idx + 1} average score: {avg_group_score:.4f} (from {len(round_scores)} rounds)")
                        
                        # Apply ranking-based ELO bonuses for this group
                        if len(group_scores) >= 2:  # Need at least 2 miners for ranking
                            # Sort miners by their average score in this group (highest first)
                            group_rankings = sorted(
                                [(uid, sum(scores) / len(scores)) for uid, scores in group_scores.items()],
                                key=lambda x: x[1],
                                reverse=True
                            )
                            
                            # ELO bonuses for 6 miners: 1st=36, 2nd=30, 3rd=24, 4th=18, 5th=12, 6th=6
                            elo_bonuses = [36, 30, 24, 18, 12, 6]
                            
                            # Assign ELO bonuses based on rank
                            for rank, (uid, score) in enumerate(group_rankings):
                                if rank < len(elo_bonuses):
                                    elo_bonus = elo_bonuses[rank]
                                    bt.logging.info(f"Group {group_idx + 1}: Miner {uid} ranked {rank + 1} (score: {score:.2f}) -> ELO bonus: +{elo_bonus}")
                                    
                                    # Store ELO bonus for this miner in this group
                                    if uid not in miner_scores:
                                        miner_scores[uid] = []
                                    miner_scores[uid].append(elo_bonus)  # Store ELO bonus instead of raw score
                                else:
                                    bt.logging.warning(f"Group {group_idx + 1}: Miner {uid} ranked {rank + 1} but no ELO bonus available")
                        
                        total_miners_evaluated += len(working_responses)
                        
                    except Exception as e:
                        bt.logging.error(f"Failed to evaluate group {group_idx + 1}: {e}")
                        continue

                else:
                    bt.logging.warning(f"Group {group_idx + 1}: No working miners for evaluation")
            
            # Calculate total ELO bonuses for each miner
            bt.logging.info(f"Tournament evaluation summary: {total_miners_evaluated} miners evaluated across {len(all_groups)} groups")
            bt.logging.info(f"Collected ELO bonuses for {len(miner_scores)} unique miners")
            
            # Calculate total ELO bonuses and prepare for submission
            final_miner_uids = []
            final_total_elo_bonuses = []
            
            for uid, elo_bonuses in miner_scores.items():
                if elo_bonuses:  # Only include miners with valid ELO bonuses
                    total_elo_bonus = sum(elo_bonuses)
                    final_miner_uids.append(uid)
                    final_total_elo_bonuses.append(total_elo_bonus)
                    bt.logging.info(f"Miner {uid}: {len(elo_bonuses)} ELO bonuses, total: +{total_elo_bonus}")
            
            # Submit total ELO bonuses to Supabase
            if self.elo_manager and final_miner_uids and final_total_elo_bonuses:
                bt.logging.info(f"Submitting total ELO bonuses for {len(final_miner_uids)} miners")
                self.submit_elo_ratings(self.current_epoch, final_miner_uids, final_total_elo_bonuses)
                
                # Store current ELO bonuses for next cycle
                self.store_current_elo_scores(final_miner_uids, final_total_elo_bonuses)
                
                # Calculate final ELO scores and weights for this validator
                final_scores = {}
                for uid, total_elo_bonus in zip(final_miner_uids, final_total_elo_bonuses):
                    # Get previous ELO score or use default
                    previous_elo = previous_scores.get(uid, 1000)
                    final_elo = previous_elo + total_elo_bonus
                    
                    # Store final ELO score (not normalized yet)
                    if uid < len(self.metagraph.hotkeys):
                        miner_hotkey = self.metagraph.hotkeys[uid]
                        final_scores[miner_hotkey] = {
                            'final_elo': final_elo,
                            'uid': uid  # Store UID for weight calculation
                        }
                
                # Store final scores for this validator
                if final_scores and self.elo_manager:
                    validator_hotkey = self.metagraph.hotkeys[self.validator_uid] if self.validator_uid < len(self.metagraph.hotkeys) else None
                    if validator_hotkey:
                        self.elo_manager.store_validator_final_scores(self.current_epoch, validator_hotkey, final_scores)
                        
                        # Calculate normalized weights (0 to 1) based on ELO scores
                        self.calculate_and_set_elo_weights(final_scores)
                        
                        bt.logging.info(f"✅ Epoch {self.current_epoch} ELO weights calculated and set")
                    else:
                        bt.logging.error(f"❌ Validator hotkey not available for UID {self.validator_uid}")
            else:
                bt.logging.warning("No ELO data collected - no miners were successfully evaluated")
            
            # Update epoch counter after all groups processed
            self.current_epoch += 1
            self.evaluation_round = 0
            
            # Check if we should refresh scores (every 10 epochs)
            if self.current_epoch % 10 == 0:
                self.refresh_elo_scores()
            
            bt.logging.info(f"Tournament evaluation complete. Epoch {self.current_epoch - 1} finished. Total miners evaluated: {total_miners_evaluated}")
            bt.logging.info(f"Current epoch: {self.current_epoch}")
            
            # Simple epoch management - wait for next cycle
            bt.logging.info(f"🔄 Waiting for next validation cycle...")
            await asyncio.sleep(60)  # Wait 1 minute before next cycle
            
        except Exception as e:
            bt.logging.error(f"Error during tournament forward: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))

    def get_epoch_status(self):
        """Get current epoch status"""
        if not self.elo_manager:
            return None
            
        try:
            status = self.elo_manager.get_epoch_status(self.current_epoch)
            bt.logging.info(f"📊 Epoch {self.current_epoch} status: {status}")
            return status
        except Exception as e:
            bt.logging.error(f"Error getting epoch status: {e}")
            return None

    def set_weights(self):
        """
        Sets validator weights based on ELO rankings from tournament evaluation.
        Overrides the base validator method to use tournament-based weights.
        """
        try:
            # Priority: ELO weights > Default weights
            if hasattr(self, 'elo_weights') and self.elo_weights is not None:
                bt.logging.info("✅ Using ELO-based weights for blockchain weight assignment")
                raw_weights = self.elo_weights.numpy()
            else:
                bt.logging.warning("⚠️ No ELO weights available, falling back to default weights")
                # Fallback to default weight setting
                super().set_weights()
                return
            
            # Log ELO weight setting process
            bt.logging.debug(f"ELO weights before processing: {raw_weights}")
            bt.logging.debug(f"ELO weights sum: {raw_weights.sum()}")
            bt.logging.debug(f"ELO weights max: {raw_weights.max()}")
            bt.logging.debug(f"ELO weights min: {raw_weights.min()}")
            
            # Show weight distribution for debugging
            non_zero_weights = raw_weights[raw_weights > 0]
            if len(non_zero_weights) > 0:
                bt.logging.info(f"📊 Weight distribution: {len(non_zero_weights)} miners with weights")
                bt.logging.info(f"   Min weight: {non_zero_weights.min():.4f}")
                bt.logging.info(f"   Max weight: {non_zero_weights.max():.4f}")
                bt.logging.info(f"   Weight sum: {non_zero_weights.sum():.4f}")
            
            # Log weight setting process to Supabase if available
            if self.supabase_mode and self.supabase:
                try:
                    weight_debug_data = {
                        "elo_weights": raw_weights.tolist(),
                        "elo_weights_sum": float(raw_weights.sum()),
                        "elo_weights_max": float(raw_weights.max()),
                        "elo_weights_min": float(raw_weights.min()),
                        "step": self.step,
                        "epoch": self.current_epoch
                    }
                    
                    # Monitoring removed - simplified system
                    bt.logging.debug("ELO weights set successfully")
                except Exception as e:
                    bt.logging.warning(f"Error reporting ELO weight data to Supabase: {e}")
            
            bt.logging.debug("raw_weight_uids", self.metagraph.uids)
            
            # Process the raw weights to final_weights via subtensor limitations.
            (
                processed_weight_uids,
                processed_weights,
            ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids=self.metagraph.uids,
                weights=raw_weights,
                netuid=self.config.netuid,
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )
            bt.logging.debug("processed_weights", processed_weights)
            bt.logging.debug("processed_weight_uids", processed_weight_uids)

            # Convert to uint16 weights and uids.
            (
                uint_uids,
                uint_weights,
            ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
                uids=processed_weight_uids, weights=processed_weights
            )
            bt.logging.debug("uint_weights", uint_weights)
            bt.logging.debug("uint_uids", uint_uids)

            # Set the weights on chain via our subtensor connection.
            result = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_finalization=False,
                wait_for_inclusion=False
            )
            bt.logging.info(f"Set ELO-based weights: {result}")
            
        except Exception as e:
            bt.logging.error(f"Error setting ELO-based weights: {e}")
            bt.logging.warning("Falling back to default weight setting")
            # Fallback to default weight setting
            super().set_weights()

    def run(self):
        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")
        
        # Set validator UID for ELO sync
        if self.validator_uid is None:
            try:
                self.validator_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
                bt.logging.info(f"Validator UID set to: {self.validator_uid}")
                
                # Set hotkey in ELO sync manager if available
                if self.elo_manager:
                    validator_hotkey = self.wallet.hotkey.ss58_address
                    self.elo_manager.set_validator_hotkey(validator_hotkey)
                    bt.logging.info(f"Validator hotkey set in ELO sync manager: {validator_hotkey}")
                    
            except ValueError:
                bt.logging.warning("Validator not found in metagraph")
        
        self.axon.start()
        bt.logging.info("Axon started and ready to handle OfficialSynapse requests.")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync(self.supabase)

                self.step += 1

                # Sleep interval before the next iteration.
                time.sleep(self.config.neuron.search_request_interval)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and exit. (restart by pm2)
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
            self.should_exit = True

    async def run_async(self):
        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")
        self.axon.start()
        bt.logging.info("Axon started and ready to handle OfficialSynapse requests.")

        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")
                await self.concurrent_forward()

                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()
                self.step += 1
                await asyncio.sleep(self.config.neuron.search_request_interval)

        except asyncio.CancelledError:
            self.axon.stop()
            bt.logging.success("Validator cancelled.")
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
            self.should_exit = True

    def print_info(self):
        metagraph = self.metagraph
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        log = (
            "Validator | "
            f"Step:{self.step} | "
            f"UID:{self.uid} | "
            f"Block:{self.block} | "
            f"Stake:{metagraph.S[self.uid]} | "
            f"VTrust:{metagraph.Tv[self.uid]} | "
            f"Dividend:{metagraph.D[self.uid]} | "
            f"Emission:{metagraph.E[self.uid]}"
        )
        bt.logging.info(log)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    intialization = True
    with Validator() as validator:
        while True:
            if validator.should_exit:
                bt.logging.warning("Ending validator...")
                break

            # wait before the first print_info, to avoid websocket connection race condition
            if intialization:
                time.sleep(60 * 5)
                intialization = False

            time.sleep(60)
            validator.print_info()