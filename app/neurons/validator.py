import os
import time
from datetime import datetime
from traceback import print_exception

import bittensor as bt
import openai
from dotenv import load_dotenv
import asyncio

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
            miner_uids = get_miners_uids(self, k=self.config.neuron.sample_size)

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


            # Log the results for monitoring purposes.
            bt.logging.trace(f"Received responses: {responses}")

            rewards, reference_response = self.evaluator.evaluate(query, responses)
            for idx, uid in enumerate(miner_uids):
                print(f"AXON FOR WALLET: {self.metagraph.axons[uid]}")

                print(f"AXON FOR WALLET DICT: {dict(self.metagraph.axons[uid])}")
                if rewards[idx] < BAD_MINER_THRESHOLD:
                    self.bad_miners_register[(uid,
                                              self.metagraph.axons[uid].wallet.hotkey)] = self.bad_miners_register.get(uid, 0) + 1

                if self.bad_miners_register.get((uid,
                                                 self.metagraph.axons[uid].wallet.hotkey), 0) > MAX_BAD_RESPONSES_TOLERANCE:
                    rewards[idx] = 0

            self.synthetics_generator.update_dialog(synthetic_dialog['dialog_id'], reference_response)
            
            rewards = rewards / (rewards.max() + 1e-5)

            bt.logging.info(f"Scored responses: {rewards} for {miner_uids}")
            self.update_scores(rewards, miner_uids)
            if self.supabase_mode:
                try:
                    #temporary measure to debug the incentive distribution situation
                    self.supabase.table('monitoring').insert({
                        "uid": self.uid,
                        "operation": "update_scores",
                        "data": {"scores": self.scores.tolist()},
                    }).execute()
                except Exception as e:
                    bt.logging.warning(f"Error reporting scores: {e}")
            bt.logging.info(f"All rewards: {rewards}")
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