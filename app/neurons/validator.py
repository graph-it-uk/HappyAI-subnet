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
from app.chain.evaluation.simple_elo_sync import SimpleELOSync
from app.chain.protocol import CompletionSynapse
from app.chain.synthetics.generator import SyntheticsGenerator

from app.chain.utils.uids import get_miners_uids
from app.chain.worker import Worker
from supabase import create_client

BAD_MINER_THRESHOLD = -5
MAX_BAD_RESPONSES_TOLERANCE = 2




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
        
        # Initialize ELO sync manager if Supabase is enabled
        self.elo_sync_manager = None
        if self.supabase_mode and self.supabase:
            try:
                supabase_url = os.environ.get("SUPABASE_URL")
                supabase_key = os.environ.get("SUPABASE_KEY")
                if supabase_url and supabase_key:
                    self.elo_sync_manager = SimpleELOSync(supabase_url, supabase_key)
                    bt.logging.info("✅ Simple ELO sync initialized successfully")
                else:
                    bt.logging.warning("⚠️ Supabase credentials not found, ELO sync disabled")
            except Exception as e:
                bt.logging.error(f"❌ Failed to initialize ELO sync manager: {e}")
                self.elo_sync_manager = None
        
        # Initialize evaluator
        self.evaluator = Evaluator(llm_client, self.worker)
        bt.logging.info("✅ Evaluator initialized")
        
        # Simple epoch management
        self.current_epoch = 0
        self.epoch_start_time = time.time()
        self.epoch_duration = 600  # 10 minutes per epoch
        self.spec_version = 1  # Version for weight setting
        self.tournament_group_size = 6  # Default group size for tournaments
        
        self.bad_miners_register = {}
        self.supabase_mode = os.environ.get("SUPABASE_MODE", "False").lower() == "true"
        self.supabase = None
        if self.supabase_mode:
            self.supabase = create_client(
                supabase_url=os.environ.get("SUPABASE_URL"),
                supabase_key=os.environ.get("SUPABASE_KEY")
            )
        
        # Load banned miners
        self.banned_coldkeys = set()
        self.banned_uids_cache = []
        self.load_banned_miners()
        self.global_miner_index = 0
        
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
    
    def submit_elo_ratings(self, epoch: int, miner_uids: List[int], rewards: List[float]):
        """
        Submit ELO ratings for evaluated miners.
        No thresholds, no waiting - just submit results.
        
        Args:
            epoch: Current epoch number
            miner_uids: List of miner UIDs evaluated
            rewards: List of reward scores for each miner
        """
        if not self.elo_sync_manager:
            bt.logging.debug("No ELO sync manager available")
            return
            
        if not self.validator_uid:
            bt.logging.warning("Validator UID not set, cannot submit ratings")
            return
            
        try:
            # Submit ELO ratings for evaluated miners
            for uid, reward in zip(miner_uids, rewards):
                if reward > 0:  # Only submit positive ratings
                    elo_rating = int(reward * 1000)  # Convert reward to ELO scale
                    success = self.elo_sync_manager.submit_elo_rating(epoch, uid, elo_rating, self.validator_uid)
                    if success:
                        bt.logging.debug(f"✅ ELO rating {elo_rating} submitted for miner {uid}")
                    else:
                        bt.logging.warning(f"⚠️ Failed to submit ELO rating for miner {uid}")
                        
        except Exception as e:
            bt.logging.error(f"Error submitting ELO ratings: {e}")

    def get_tournament_groups(self, group_size: int = 6, exclude: List[int] = None, specified_miners: List[int] = None) -> List[List[int]]:
        """
        Get all tournament groups for this epoch.
        
        Args:
            group_size (int): Number of miners per group
            exclude (List[int]): List of UIDs to exclude
            specified_miners (List[int]): List of specific miners to use
            
        Returns:
            List[List[int]]: All tournament groups for this epoch
        """
        # Set current epoch and evaluation round for proper grouping
        self.current_epoch = self.current_epoch
        self.evaluation_round = 0
        
        return self.tournament_group_miners(group_size, exclude, specified_miners)

    async def forward(self):
        """
        Validator forward pass with tournament evaluation. Consists of:
        - Generating the query
        - Creating tournament groups
        - Evaluating each group
        - Updating ELO ratings
        - Setting weights based on ELO rankings
        """
        try:
            # Get banned miner UIDs to exclude from selection
            banned_uids = self.get_banned_miner_uids()
            
            # Get all tournament groups for this epoch
            all_groups = self.get_tournament_groups(group_size=self.tournament_group_size, exclude=banned_uids)
            bt.logging.info(f"Created {len(all_groups)} tournament groups for epoch {self.current_epoch}")
            
            if not all_groups:
                bt.logging.warning("No tournament groups created, skipping evaluation")
                return
            
            # Submit ELO ratings for this evaluation round
            if self.elo_sync_manager:
                self.submit_elo_ratings(self.current_epoch, miner_uids, rewards)
            
            # Generate synthetic query for all groups
            synthetic_dialog = await self.synthetics_generator.generate()
            query = CompletionSynapse(request_id = int(datetime.now().timestamp()*1000),
                                    messages = synthetic_dialog['messages'][:-1],
                                    user_input = synthetic_dialog['messages'][-1].content)
            bt.logging.info('Generated synthetic query for tournament evaluation')
            
            # Process each tournament group
            total_miners_evaluated = 0
            for group_idx, miner_uids in enumerate(all_groups):
                bt.logging.info(f"Evaluating tournament group {group_idx + 1}/{len(all_groups)}: {miner_uids}")
                
                # Query miners in this group
                responses = await self.dendrite(
                    axons=[self.metagraph.axons[uid] for uid in miner_uids],
                    synapse=query,
                    deserialize=True,
                    timeout=20,
                )
                
                # Filter working responses
                working_miner_indices = []
                working_responses = []
                working_uids = []
                
                for idx, (uid, response) in enumerate(zip(miner_uids, responses)):
                    has_response = bool(response and hasattr(response, 'response') and response.response)
                    if has_response:
                        working_miner_indices.append(idx)
                        working_responses.append(response.response)
                        working_uids.append(uid)
                
                bt.logging.info(f"Group {group_idx + 1}: {len(working_responses)}/{len(miner_uids)} working miners")
                
                # Evaluate working miners in tournament
                if working_responses and len(working_responses) >= 2:  # Need at least 2 miners for tournament
                    try:
                        # Update evaluation round for this group
                        self.evaluation_round = group_idx
                        
                        # Evaluate using tournament system
                        working_rewards, _ = self.evaluator.evaluate(query, working_responses, working_uids)
                        
                        bt.logging.info(f"Group {group_idx + 1} tournament results: {working_rewards}")
                        total_miners_evaluated += len(working_responses)
                        
                    except Exception as e:
                        bt.logging.error(f"Failed to evaluate group {group_idx + 1}: {e}")
                        continue
                else:
                    bt.logging.warning(f"Group {group_idx + 1}: Insufficient working miners for tournament evaluation")
            
            # Update epoch counter after all groups processed
            self.current_epoch += 1
            self.evaluation_round = 0
            
            bt.logging.info(f"Tournament evaluation complete. Epoch {self.current_epoch - 1} finished. Total miners evaluated: {total_miners_evaluated}")
            
            # Simple epoch finalization - no waiting, no thresholds
            if self.elo_sync_manager:
                completed_epoch = self.current_epoch - 1
                bt.logging.info(f"🎯 Finalizing epoch {completed_epoch}")
                
                # Finalize epoch and get consensus
                consensus_ratings = self.elo_sync_manager.finalize_epoch(completed_epoch)
                
                if consensus_ratings:
                    bt.logging.info(f"✅ Epoch {completed_epoch} finalized with {len(consensus_ratings)} miners")
                    
                    # Push consensus to Bittensor
                    success = self.elo_sync_manager.push_to_bittensor(completed_epoch)
                    if success:
                        bt.logging.info(f"🚀 Epoch {completed_epoch} consensus pushed to Bittensor")
                    else:
                        bt.logging.warning(f"⚠️ Failed to push epoch {completed_epoch} to Bittensor")
                else:
                    bt.logging.warning(f"⚠️ No consensus available for epoch {completed_epoch}")
                                    # Fallback to local weights
                self.set_weights_from_elo()
            else:
                # No sync manager, use local weights
                bt.logging.info("🔄 No ELO sync manager, using local weights")
                self.set_weights_from_elo()
            
            # Check if epoch should end
            current_time = time.time()
            if current_time - self.epoch_start_time >= self.epoch_duration:
                # Epoch ended, increment and reset
                self.current_epoch += 1
                self.epoch_start_time = current_time
                bt.logging.info(f"🔄 New epoch started: {self.current_epoch}")
                
                # Log epoch status
                if self.elo_sync_manager:
                    self.get_epoch_status()
                    # Cleanup old epochs from database
                    self.elo_sync_manager.cleanup_old_epochs(keep_last=10)
            else:
                # Sleep until epoch ends
                time_until_epoch_end = self.epoch_duration - (current_time - self.epoch_start_time)
                time.sleep(min(60, time_until_epoch_end))  # Sleep in 1-minute chunks
            
        except Exception as e:
            bt.logging.error(f"Error during tournament forward: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))

    def set_weights_from_consensus(self, consensus_ratings: Dict[int, Dict]):
        """
        Set blockchain weights based on consensus ELO ratings from Supabase.
        
        Args:
            consensus_ratings: Dict mapping miner_uid to consensus data
        """
        if not consensus_ratings:
            bt.logging.warning("No consensus ratings available")
            return
        
        try:
            # Create weight tensor based on consensus ELO ratings
            weights = torch.zeros(self.metagraph.n.item())
            
            # Set weights based on consensus ratings
            for uid, consensus_data in consensus_ratings.items():
                if uid < len(weights):
                    # Use the consensus weight directly (already normalized 0-1)
                    weights[uid] = consensus_data['final_weight']
                    bt.logging.debug(f"Miner {uid}: Consensus ELO {consensus_data['final_elo']}, Weight {consensus_data['final_weight']:.4f}")
            
            # Normalize weights to sum to 1.0
            if weights.sum() > 0:
                weights = weights / weights.sum()
            
            bt.logging.info(f"Set weights from consensus ratings. Miners: {len(consensus_ratings)}")
            bt.logging.info(f"Weight distribution: min={weights.min():.4f}, max={weights.max():.4f}, sum={weights.sum():.4f}")
            
            # Store weights for later use
            self.consensus_weights = weights
            
        except Exception as e:
            bt.logging.error(f"Error setting weights from consensus: {e}")
            # Fallback to local ELO
            self.set_weights_from_elo()
    
    def get_epoch_status(self):
        """Get current epoch status"""
        if not self.elo_sync_manager:
            return None
            
        try:
            status = self.elo_sync_manager.get_epoch_status(self.current_epoch)
            bt.logging.info(f"📊 Epoch {self.current_epoch} status: {status}")
            return status
        except Exception as e:
            bt.logging.error(f"Error getting epoch status: {e}")
            return None
    
    def set_weights_from_elo(self):
        """
        Set blockchain weights based on local ELO scores.
        This method should be called during sync to update weights.
        """
        try:
            # Check if evaluator and ELO system are available
            if not hasattr(self, 'evaluator') or not hasattr(self.evaluator, 'elo_system'):
                bt.logging.warning("No ELO system available, cannot set weights")
                return
            
            # Get ELO rankings from the ELO system
            elo_rankings = self.evaluator.elo_system.get_rankings()
            
            if not elo_rankings:
                bt.logging.warning("No ELO rankings available, cannot set weights")
                return
            
            # Create weight tensor based on ELO rankings
            weights = torch.zeros(self.metagraph.n.item())
            
            # Set weights based on ELO ranking (higher ELO = higher weight)
            for rank, (uid, elo_score) in enumerate(elo_rankings):
                if uid < len(weights):
                    # Weight formula: higher ELO gets higher weight
                    # Normalize ELO scores to 0-1 range for weights
                    if elo_rankings:
                        max_elo = max(score for _, score in elo_rankings)
                        min_elo = min(score for _, score in elo_rankings)
                        
                        if max_elo > min_elo:
                            normalized_weight = (elo_score - min_elo) / (max_elo - min_elo)
                        else:
                            normalized_weight = 0.5  # All miners get equal weight if ELO is same
                        
                        weights[uid] = normalized_weight
            
            # Normalize weights to sum to 1.0
            if weights.sum() > 0:
                weights = weights / weights.sum()
            
            bt.logging.info(f"Set weights from ELO rankings. Top 5 miners: {elo_rankings[:5]}")
            bt.logging.info(f"Weight distribution: min={weights.min():.4f}, max={weights.max():.4f}, sum={weights.sum():.4f}")
            
            # Store weights for later use (they will be applied during sync)
            self.elo_weights = weights
            
        except Exception as e:
            bt.logging.error(f"Error setting weights from ELO: {e}")

    def set_weights(self):
        """
        Sets validator weights based on ELO rankings from tournament evaluation.
        Overrides the base validator method to use tournament-based weights.
        """
        try:
            # Priority: Consensus weights > ELO weights > Default weights
            if hasattr(self, 'consensus_weights') and self.consensus_weights is not None:
                bt.logging.info("✅ Using consensus-based weights for blockchain weight assignment")
                raw_weights = self.consensus_weights.numpy()
            elif hasattr(self, 'elo_weights') and self.elo_weights is not None:
                bt.logging.info("🔄 Using local ELO-based weights for blockchain weight assignment")
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
                wait_for_inclusion=False,
                version_key=self.spec_version,
            )
            bt.logging.info(f"Set ELO-based weights: {result}")
            
        except Exception as e:
            bt.logging.error(f"Error setting ELO-based weights: {e}")
            bt.logging.warning("Falling back to default weight setting")
            # Fallback to default weight setting
            super().set_weights()

    def test_tournament_integration(self):
        """
        Test method to verify tournament integration works correctly.
        This should be called before running the full validator.
        """
        try:
            bt.logging.info("Testing tournament integration...")
            
            # Test 1: Tournament group creation
            test_groups = self.get_tournament_groups(group_size=6, exclude=[])
            bt.logging.info(f"✓ Tournament groups created: {len(test_groups)} groups")
            
            # Test 2: ELO system availability
            if hasattr(self.evaluator, 'elo_system') and self.evaluator.elo_system:
                bt.logging.info("✓ ELO system available")
            else:
                bt.logging.error("✗ ELO system not available")
                return False
            
            # Test 3: Weight setting capability
            try:
                self.set_weights_from_elo()
                bt.logging.info("✓ ELO weight setting works")
            except Exception as e:
                bt.logging.error(f"✗ ELO weight setting failed: {e}")
                return False
            
            bt.logging.success("Tournament integration test passed!")
            return True
            
        except Exception as e:
            bt.logging.error(f"Tournament integration test failed: {e}")
            return False

    def run(self):
        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")
        
        # Set validator UID for ELO sync
        if self.validator_uid is None:
            try:
                self.validator_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
                bt.logging.info(f"Validator UID set to: {self.validator_uid}")
                
                # Set UID in ELO sync manager if available
                if self.elo_sync_manager:
                    self.elo_sync_manager.set_validator_uid(self.validator_uid)
                    
            except ValueError:
                bt.logging.warning("Validator not found in metagraph")
        
        # Test tournament integration before starting
        if not self.test_tournament_integration():
            bt.logging.error("Tournament integration test failed. Exiting.")
            return
        
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