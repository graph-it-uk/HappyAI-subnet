
import json
import os
import time


import bittensor as bt

from app.chain.base.miner import BaseMinerNeuron

from dotenv import load_dotenv

from app.chain.protocol import CompletionSynapse
from app.chain.worker import Worker

load_dotenv()

class Miner(BaseMinerNeuron):


    def __init__(self):
        super(Miner, self).__init__()
        self.worker = Worker(worker_url=os.environ["WORKER_URL"], worker_port=os.environ["WORKER_PORT"])
        
        # Get my own coldkey for ban checking
        self.my_coldkey = self.wallet.coldkey.ss58_address
        bt.logging.info(f"🔑 My coldkey: {self.my_coldkey}")
        
        # Initial ban check
        self.check_if_banned()

    async def forward_completion(
        self, query: CompletionSynapse
    ) -> CompletionSynapse:
        bt.logging.trace(f'Processing request: {query.request_id}')
        request_data = {
            'request_id': query.request_id,
            'user_input': {'sender': 'user', 'text': query.user_input},
            'dialog': [{'sender': message.role, 'text': message.content} for message in query.messages],
            'chat_llm_model': 'main'
        }
        response = self.worker.process_request(request_data)
        assistant_message = json.loads(response.text)['assistant_message']
        bt.logging.trace(f'Assistant message: {assistant_message}')
        query.results = assistant_message
        return query

    def check_if_banned(self):
        """Check if this miner is banned by validators"""
        try:
            with open('banned_miners.json', 'r') as f:
                ban_config = json.load(f)
                banned_coldkeys = set(ban_config.get('banned_coldkeys', []))
                
                if self.my_coldkey in banned_coldkeys:
                    # Get ban reason if available
                    ban_reasons = ban_config.get('ban_reasons', {})
                    reason = ban_reasons.get(self.my_coldkey, "No reason specified")
                    
                    bt.logging.error("🚫" + "="*80)
                    bt.logging.error("🚫 ATTENTION: THIS MINER IS BANNED!")
                    bt.logging.error(f"🚫 Your coldkey: {self.my_coldkey}")
                    bt.logging.error(f"🚫 Ban reason: {reason}")
                    bt.logging.error("🚫 Validators will NOT send you requests!")
                    bt.logging.error("🚫 Consider stopping this miner until ban is resolved.")
                    bt.logging.error("🚫" + "="*80)
                    return True
                else:
                    bt.logging.info(f"✅ Miner status: NOT BANNED (coldkey checked)")
                    return False
                    
        except FileNotFoundError:
            bt.logging.debug("📋 No banned_miners.json found - assuming no bans")
            return False
        except Exception as e:
            bt.logging.warning(f"⚠️ Could not check ban status: {e}")
            return False


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        miner_hotkey = miner.wallet.hotkey.ss58_address
        bt.logging.trace(f"My Miner hotkey: {miner_hotkey}")
        time.sleep(120)
        
        while True:
            time.sleep(3600)  # Wait for 1 hour
            is_banned = miner.check_if_banned()
            if is_banned:
                bt.logging.warning("🚫 Still banned - consider stopping this miner to save resources")
            else:
                bt.logging.debug("✅ Ban check: Still operating normally")

