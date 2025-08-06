import os
import time
import json
from datetime import datetime
from traceback import print_exception

import bittensor as bt
import openai
from dotenv import load_dotenv
import asyncio
import torch

from app.chain.base.validator import BaseValidatorNeuron
from app.chain.evaluation.evaluator import Evaluator
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
        self.evaluator = Evaluator(llm_client, self.worker)
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

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        try:
            # Get banned miner UIDs to exclude from selection
            banned_uids = self.get_banned_miner_uids()
            
            # Select miners excluding banned ones
            miner_uids = get_miners_uids(self, k=self.config.neuron.sample_size, exclude=banned_uids)
            bt.logging.warning(f"DEBUG: Selected miner UIDs: {miner_uids}")
            if banned_uids:
                bt.logging.warning(f"DEBUG: Excluded banned UIDs: {banned_uids}")
            
            synthetic_dialog = await self.synthetics_generator.generate()
            query = CompletionSynapse(request_id = int(datetime.now().timestamp()*1000),
                                    messages = synthetic_dialog['messages'][:-1],
                                    user_input = synthetic_dialog['messages'][-1].content)
            bt.logging.trace('Generated synthetic query!')
            bt.logging.trace(f'query: {query}')
            
            # The dendrite client queries the network.
            responses = await self.dendrite(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=query,
                deserialize=True,
                timeout=20,
            )
            bt.logging.trace('Sent synapse to miners!')
            bt.logging.trace(f'Miners: {miner_uids}')
            bt.logging.trace(f'Responses: {responses}')

            bt.logging.warning(f"DEBUG: Got {len(responses)} responses")
            for i, (uid, response) in enumerate(zip(miner_uids, responses)):
                has_response = bool(response and hasattr(response, 'response') and response.response)
                bt.logging.warning(f"DEBUG: Miner {uid}: response_exists={has_response}")

            # Log the results for monitoring purposes.
            bt.logging.trace(f"Received responses: {responses}")
            
            # Pre-filter non-working miners to save evaluation costs
            working_miner_indices = []
            non_working_miner_indices = []
            
            for idx, uid in enumerate(miner_uids):
                axon_ip = self.metagraph.axons[uid].ip
                if axon_ip == "0.0.0.0" or axon_ip == "127.0.0.1" or axon_ip == "localhost":
                    non_working_miner_indices.append(idx)
                    bt.logging.warning(f"DEBUG: Non-working miner UID {uid} (IP: {axon_ip})")
                else:
                    working_miner_indices.append(idx)
                    bt.logging.warning(f"DEBUG: Working miner UID {uid} (IP: {axon_ip})")
            
            bt.logging.warning(f"DEBUG: Working={len(working_miner_indices)}, Non-working={len(non_working_miner_indices)}")
            
            # Only evaluate working miners
            working_responses = [responses[i] for i in working_miner_indices] if working_miner_indices else []
            
            bt.logging.warning(f"DEBUG: Evaluating {len(working_responses)} working responses")
            
            if working_responses:
                working_rewards, reference_response = self.evaluator.evaluate(query, working_responses)
                bt.logging.warning(f"DEBUG: working_rewards from evaluator: {working_rewards}")
                bt.logging.warning(f"DEBUG: working_rewards min={working_rewards.min()}, max={working_rewards.max()}")
            else:
                working_rewards = torch.tensor([])
                reference_response = None
                bt.logging.warning(f"DEBUG: No working responses, empty tensor")
            
            # Create final scores array with -1 for non-working miners
            rewards = torch.zeros(len(miner_uids))
            bt.logging.warning(f"DEBUG: Initialized rewards tensor: {rewards}")
            
            # Add normalized working miner scores (already normalized in evaluator)
            if working_rewards.numel() > 0:
                for i, working_idx in enumerate(working_miner_indices):
                    rewards[working_idx] = working_rewards[i]
                    bt.logging.warning(f"DEBUG: Set working miner {miner_uids[working_idx]} reward = {working_rewards[i]}")
            
            # Add -1 for non-working miners
            for idx in non_working_miner_indices:
                rewards[idx] = -1.0
                bt.logging.warning(f"DEBUG: Set non-working miner {miner_uids[idx]} reward = -1.0")
            
            bt.logging.warning(f"DEBUG: Raw rewards after assignment: {rewards}")
            
            # Apply bad miner penalties (if any)
            # Replace this entire section in your validator forward() method:

            # Apply bad miner penalties (if any)
            penalties_applied = 0
            for idx, uid in enumerate(miner_uids):
                old_reward = rewards[idx].clone()
                
                if rewards[idx] < BAD_MINER_THRESHOLD:
                    hotkey = self.metagraph.axons[uid].hotkey  
                    current_count = self.bad_miners_register.get((uid, hotkey), 0)
                    self.bad_miners_register[(uid, hotkey)] = current_count + 1
                    bt.logging.warning(f"DEBUG: Miner {uid} flagged as bad (reward {rewards[idx]} < {BAD_MINER_THRESHOLD}). Count: {current_count + 1}")
                
                hotkey = self.metagraph.axons[uid].hotkey 
                bad_count = self.bad_miners_register.get((uid, hotkey), 0)
                if bad_count > MAX_BAD_RESPONSES_TOLERANCE:
                    rewards[idx] = -0.8
                    penalties_applied += 1
                    bt.logging.warning(f"DEBUG: Applied -0.8 penalty to miner {uid} (bad_count={bad_count})")
                
                if old_reward != rewards[idx]:
                    bt.logging.warning(f"DEBUG: Miner {uid} reward changed: {old_reward} -> {rewards[idx]}")
            
            bt.logging.warning(f"DEBUG: After penalties ({penalties_applied} applied): {rewards}")
            
            self.synthetics_generator.update_dialog(synthetic_dialog['dialog_id'], reference_response)
            
            bt.logging.debug(f"Final scores before Bittensor normalization: {rewards}")
            bt.logging.debug(f"Rewards max: {rewards.max()}")
            bt.logging.debug(f"Rewards min: {rewards.min()}")
            
            # Final normalization for Bittensor (handles negative values)
            pre_norm_rewards = rewards.clone()
            bt.logging.warning(f"DEBUG: Before final normalization: min={rewards.min()}, max={rewards.max()}")
            
            if rewards.max() > 0:
                rewards = rewards / (rewards.max() + 1e-5)
                bt.logging.warning(f"DEBUG: Applied normalization (divided by {pre_norm_rewards.max()})")
            else:
                bt.logging.error(f"DEBUG: WARNING! All rewards <= 0, no normalization applied")
                bt.logging.error(f"DEBUG: This might cause problems in update_scores!")
            
            bt.logging.warning(f"DEBUG: Final rewards: {rewards}")
            bt.logging.warning(f"DEBUG: Final min={rewards.min()}, max={rewards.max()}")
            
            # Log the mapping
            for i, uid in enumerate(miner_uids):
                miner_type = "WORKING" if i in working_miner_indices else "NON-WORKING"
                bt.logging.warning(f"DEBUG: UID {uid} ({miner_type}) -> reward {rewards[i]}")
            
            # Check UID mapping before update
            bt.logging.warning(f"DEBUG: UID MAPPING CHECK:")
            bt.logging.warning(f"DEBUG: miner_uids: {miner_uids}")
            bt.logging.warning(f"DEBUG: metagraph total size: {len(self.metagraph.axons)}")
            bt.logging.warning(f"DEBUG: self.scores size: {len(self.scores)}")
            
            for uid in miner_uids:
                if uid >= len(self.scores):
                    bt.logging.error(f"DEBUG: CRITICAL: UID {uid} >= scores length {len(self.scores)}")
                
                if uid < len(self.scores):
                    current_score = self.scores[uid]
                    bt.logging.warning(f"DEBUG: UID {uid}: current_score = {current_score:.6f}")
            
            bt.logging.warning(f"DEBUG: Calling update_scores with rewards: {rewards}")
            bt.logging.warning(f"DEBUG: Calling update_scores with UIDs: {miner_uids}")
            
            self.update_scores(rewards, miner_uids)
            
            # Check scores after update
            bt.logging.warning(f"DEBUG: After update_scores, current self.scores:")
            for uid in miner_uids:
                if uid < len(self.scores):
                    bt.logging.warning(f"DEBUG: UID {uid}: score = {self.scores[uid]:.6f}")
            
            if self.supabase_mode:
                try:
                    # Enhanced debugging data for Supabase
                    debug_data = {
                        "scores": self.scores.tolist(),
                        "raw_rewards": rewards.tolist() if hasattr(rewards, 'tolist') else rewards,
                        "miner_uids": miner_uids,
                        "rewards_max": float(rewards.max()),
                        "rewards_min": float(rewards.min()),
                        "non_zero_scores_count": int((self.scores > 0).sum()),
                        "scores_sum": float(self.scores.sum()),
                        "alpha": self.config.neuron.moving_average_alpha,
                        "step": self.step
                    }
                    
                    self.supabase.table('monitoring').insert({
                        "uid": self.uid,
                        "operation": "update_scores",
                        "data": debug_data,
                    }).execute()
                except Exception as e:
                    bt.logging.warning(f"Error reporting scores: {e}")
            
            bt.logging.info(f"All rewards: {rewards}")
            bt.logging.debug(f"Current scores after update: {self.scores}")
            bt.logging.debug(f"Non-zero scores count: {(self.scores > 0).sum()}")
            bt.logging.debug(f"Scores sum: {self.scores.sum()}")
            time.sleep(600)
        except Exception as e:
            bt.logging.error(f"Error during forward: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))


    def run(self):
        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")
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