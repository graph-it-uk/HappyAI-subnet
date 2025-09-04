import os
import time
import json
from datetime import datetime
from traceback import print_exception
from typing import List, Dict

import requests
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
        
        # Initialize ELO sync manager (required for validator operation)
        try:
            self.elo_manager = EloManager(self)
        except Exception as e:
            bt.logging.error(f"❌ Failed to initialize ELO manager: {e}")
            raise RuntimeError(f"ELO manager initialization failed: {e}")
        
        # Initialize evaluator
        self.evaluator = Evaluator(llm_client, self.worker)
        
        # Epoch management
        self.current_epoch = 0
        self.epoch_duration = 600  # 10 minutes per epoch
        self.tournament_group_size = 6  # Default group size for tournaments
        self.num_evaluation_rounds = 3  
        self.evaluation_round = 0
        
        
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
        try:
            previous_scores = {}
            
            # Get validator hotkey
            if not hasattr(self, 'validator_uid') or self.validator_uid is None:
                bt.logging.warning("Validator UID not set yet")
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
                
                response = requests.get(
                    f"{self.api_base}/get-miner-elo",
                    params={
                        "validator_hotkey": validator_hotkey,
                        "miner_hotkey": miner_hotkey
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    if result['found']:
                        current_elo = result['elo_rating']
                        previous_scores[uid] = current_elo
                        bt.logging.debug(f"Miner {uid} ({miner_hotkey}): Current ELO score {current_elo}")
                    else:
                        previous_scores[uid] = 1000
                        bt.logging.debug(f"Miner {uid} ({miner_hotkey}): No previous score, using default 1000")
                else:
                    bt.logging.error(f"API error {response.status_code}: {response.text}")
                    previous_scores[uid] = 1000
                            
            bt.logging.info(f"Retrieved previous ELO scores for {len(previous_scores)} miners")
            return previous_scores
            
        except Exception as e:
            bt.logging.error(f"Error getting previous ELO scores: {e}")
            return {}
    
    
    

    def submit_elo_ratings(self, epoch: int, miner_uids: List[int], rewards: List[float], previous_scores: Dict[int, float] = None, miner_responses: Dict[int, str] = None):
        """
        Submit ELO ratings for evaluated miners.
        Accumulates new scores with previous scores.
        
        Args:
            epoch: Current epoch number
            miner_uids: List of miner UIDs evaluated
            rewards: List of reward scores for each miner
            previous_scores: Dict of previous ELO scores (optional, will fetch if not provided)
            miner_responses: Dict mapping miner UID to their response content (optional)
        """
        if not self.validator_uid:
            bt.logging.warning("Validator UID not set, cannot submit ratings")
            return
            
        try:
            # Use provided previous scores or fetch them if not provided
            if previous_scores is None:
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
                    
                    # Get miner response if available
                    miner_response = miner_responses.get(uid) if miner_responses else None
                    
                    if validator_hotkey:
                        success = self.elo_manager.submit_elo_rating(self.current_epoch, miner_hotkey, accumulated_elo, validator_hotkey, miner_response)
                        if success:
                            bt.logging.debug(f"✅ ELO rating {accumulated_elo} (previous: {previous_elo} + bonus: +{elo_bonus}) submitted for miner {uid} ({miner_hotkey}), Response stored: {'Yes' if miner_response else 'No'}")
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



    async def forward(self):
        """
        Validator forward pass with tournament evaluation.
        """
        try:
            # Get banned miner UIDs to exclude from selection
            banned_uids = self.get_banned_miner_uids()
            
            # Get all current miner UIDs from metagraph (excluding validators)
            all_miner_uids = [uid for uid in range(self.metagraph.n) if not self.metagraph.validator_permit[uid]]
            bt.logging.info(f"Found {len(all_miner_uids)} miners in metagraph")
            
            # Load previous ELO scores for all miners
            previous_scores = self.get_previous_elo_scores(all_miner_uids)
            bt.logging.info(f"Loaded previous ELO scores for {len(previous_scores)} miners")
            
            # Generate synthetic query ONCE for all miners
            synthetic_dialog = await self.synthetics_generator.generate()
            query = CompletionSynapse(
                request_id=int(datetime.now().timestamp() * 1000),
                messages=synthetic_dialog['messages'][:-1],
                user_input=synthetic_dialog['messages'][-1].content
            )
            bt.logging.info('Generated synthetic query for tournament evaluation')
            
            # Query ALL miners ONCE with the synthetic query
            bt.logging.info(f"Querying {len(all_miner_uids)} miners with synthetic query...")
            all_responses = await self.dendrite(
                axons=[self.metagraph.axons[uid] for uid in all_miner_uids],
                synapse=query,
                deserialize=True,
                timeout=45,
            )
            bt.logging.info(f"Received responses from {len(all_responses)} miners")
            
            # Filter working responses and create response mapping
            miner_response_map = {}  # uid -> response_content
            working_miner_uids = []
            
            for _, (uid, response) in enumerate(zip(all_miner_uids, all_responses)):
                response_content = None
                
                if response:
                    # Check for results attribute (correct format from miner)
                    if hasattr(response, 'results') and response.results:
                        response_content = response.results
                        bt.logging.debug(f"Miner {uid}: Valid response in 'results' attribute")
                    # Check for response attribute (fallback)
                    elif hasattr(response, 'response') and response.response:
                        response_content = response.response
                        bt.logging.debug(f"Miner {uid}: Valid response in 'response' attribute")
                    # Check if response itself is a string (direct response)
                    elif isinstance(response, str) and response.strip():
                        response_content = response
                        bt.logging.debug(f"Miner {uid}: Valid direct string response")
                    # Check if response has any content (last resort)
                    elif response and str(response).strip():
                        response_content = str(response)
                        bt.logging.debug(f"Miner {uid}: Valid response content found")
                
                if response_content:
                    miner_response_map[uid] = response_content
                    working_miner_uids.append(uid)
            
            bt.logging.info(f"Working miners: {len(working_miner_uids)}/{len(all_miner_uids)}")
            
            if len(working_miner_uids) < 2:
                bt.logging.warning("Not enough working miners for tournament evaluation (need at least 2)")
                return
            
            # Local score collection - store scores for each miner across all rounds
            miner_scores = {}  # uid -> list of scores
            
            # Run evaluation rounds with NEW groups each time, but SAME responses
            for round_num in range(self.num_evaluation_rounds):
                bt.logging.info(f"🔄 Starting evaluation round {round_num + 1}/{self.num_evaluation_rounds}")
                
                # Update evaluation round for tournament grouping
                self.evaluation_round = round_num
                
                # Create NEW tournament groups for this round
                round_groups = tournament_group_miners(self, banned_uids, working_miner_uids, self.tournament_group_size)
                bt.logging.info(f"Round {round_num + 1}: Created {len(round_groups)} new tournament groups")
                
                # Process each tournament group in this round
                for group_idx, miner_uids in enumerate(round_groups):
                    bt.logging.info(f"Round {round_num + 1}, Group {group_idx + 1}/{len(round_groups)}: {miner_uids}")
                    
                    # Get responses for this group from our stored responses (no new queries)
                    group_responses = []
                    group_working_uids = []
                    
                    for uid in miner_uids:
                        if uid in miner_response_map:
                            group_responses.append(miner_response_map[uid])
                            group_working_uids.append(uid)
                    
                    bt.logging.info(f"Round {round_num + 1}, Group {group_idx + 1}: {len(group_responses)}/{len(miner_uids)} working miners")
                    
                    # Evaluate working miners in tournament for this round
                    if group_responses and len(group_responses) >= 2:  # Need at least 2 miners for tournament
                        try:
                            # Evaluate using tournament system (this runs multiple criteria internally)
                            working_rewards = self.evaluator.evaluate(query, group_responses, group_working_uids)
                            
                            # Ensure working_rewards is a valid tensor/list
                            if working_rewards is None or len(working_rewards) != len(group_working_uids):
                                bt.logging.error(f"Round {round_num + 1}, Group {group_idx + 1}: Invalid evaluation results")
                                continue
                            
                            for uid, reward in zip(group_working_uids, working_rewards):
                                if uid not in miner_scores:
                                    miner_scores[uid] = []
                                miner_scores[uid].append(reward)
                                bt.logging.debug(f"Round {round_num + 1}, Miner {uid}: Score {reward:.4f}")
                                
                        except Exception as e:
                            bt.logging.error(f"Round {round_num + 1}, Group {group_idx + 1} failed: {e}")
                            continue
                    else:
                        bt.logging.warning(f"Round {round_num + 1}, Group {group_idx + 1}: No working miners for evaluation")
                
                bt.logging.info(f"🔄 Completed evaluation round {round_num + 1}/{self.num_evaluation_rounds}")
            

            
            # Calculate ELO bonuses based on final rubric scores, not tournament rankings
            final_miner_uids = []
            final_total_elo_bonuses = []
            
            for uid, round_scores in miner_scores.items():
                if round_scores:  # Only include miners with valid scores
                    # Calculate average score across all rounds
                    avg_score = sum(round_scores) / len(round_scores)
                    bt.logging.info(f"Miner {uid}: {len(round_scores)} rounds, average score: {avg_score:.2f}")
                    

                    elo_bonus = int(avg_score * 3.6)  # This gives max 36 for perfect score (10 * 3.6)
                    elo_bonus = max(0, min(36, elo_bonus))  # Clamp between 0 and 36
                    
                    final_miner_uids.append(uid)
                    final_total_elo_bonuses.append(elo_bonus)
                    bt.logging.info(f"Miner {uid}: Average score {avg_score:.2f} -> ELO bonus: +{elo_bonus}")
                else:
                    bt.logging.warning(f"Miner {uid}: No scores available, skipping ELO calculation")
            
            if final_miner_uids and final_total_elo_bonuses:
                bt.logging.info(f"Submitting total ELO bonuses for {len(final_miner_uids)} miners")
                
                # Apply soft forgetting to bonuses BEFORE sending to DB if it's the 10th epoch
                final_bonuses = final_total_elo_bonuses.copy()
                if self.current_epoch % 10 == 0:
                    bt.logging.info(f"🎯 Epoch {self.current_epoch}: Applying soft forgetting to ELO bonuses before DB submission")
                    base_rating = 1000
                    gamma = 0.4
                    
                    for i, (uid, bonus) in enumerate(zip(final_miner_uids, final_total_elo_bonuses)):
                        previous_elo = previous_scores.get(uid, 1000)
                        current_elo = previous_elo + bonus
                        # Apply soft forgetting: R' = μ + γ * (R - μ)
                        forgotten_elo = base_rating + gamma * (current_elo - base_rating)
                        # Convert back to bonus
                        final_bonuses[i] = forgotten_elo - previous_elo
                        bt.logging.debug(f"Miner {uid}: {current_elo} → {forgotten_elo} (bonus: {bonus} → {final_bonuses[i]})")
                
                # Submit to DB with forgotten bonuses
                self.submit_elo_ratings(self.current_epoch, final_miner_uids, final_bonuses, previous_scores, miner_response_map)
                
                # Calculate final ELO scores and weights for this validator
                final_scores = {}
                for uid, total_elo_bonus in zip(final_miner_uids, final_bonuses):
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
                
                # Calculate and set weights
                if final_scores:
                    validator_hotkey = self.metagraph.hotkeys[self.validator_uid] if self.validator_uid < len(self.metagraph.hotkeys) else None
                    if validator_hotkey:                        
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
            
            bt.logging.info(f"Current epoch: {self.current_epoch}")
            
            # Simple epoch management - wait for next cycle
            bt.logging.info(f"🔄 Waiting for next validation cycle...")
            await asyncio.sleep(60)  # Wait 1 minute before next cycle
            
        except Exception as e:
            bt.logging.error(f"Error during tournament forward: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))



    def set_weights(self):
        """
        Sets validator weights based on ELO rankings from tournament evaluation.
        Overrides the base validator method to use tournament-based weights.
        """
        try:
            # Check if we have ELO weights available
            if not hasattr(self, 'elo_weights') or self.elo_weights is None:
                bt.logging.warning("⚠️ No ELO weights available, falling back to default weights")
                super().set_weights()
                return
            
            bt.logging.info("✅ Using ELO-based weights for blockchain weight assignment")
            
            # Use ELO weights instead of base scores
            self.scores = self.elo_weights.clone()
            
            # Call base method to handle the actual weight setting
            super().set_weights()
            
            bt.logging.info(f"✅ ELO-based weights set successfully for {len(self.elo_weights)} miners")
            
        except Exception as e:
            bt.logging.error(f"Error setting ELO-based weights: {e}")
            bt.logging.warning("Falling back to default weight setting")
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
                self.sync()

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