"""
Simple ELO Sync System - Independent Validators
- Each validator stores their own ELO scores
- Each validator pushes their own weights to Bittensor
- No consensus, no synchronization delays
"""
import json
import time
from typing import Dict, List, Optional
import bittensor as bt
import requests

class EloManager:
    """
    Independent ELO sync - each validator works independently.
    No consensus, no waiting - just store scores and push weights.
    """
    
    def __init__(self, validator_instance=None):
        # Test connection BEFORE creating the object

        self.api_url = "http://72.60.35.80/api/validator/submit-evaluation"
        
            
        self.validator_instance = validator_instance
        self.validator_hotkey = None
    
    def set_validator_hotkey(self, validator_hotkey: str):
        """Set validator hotkey for ELO sync operations"""
        self.validator_hotkey = validator_hotkey
        bt.logging.info(f"✅ Validator hotkey set to {validator_hotkey} in ELO sync manager")
    
    def submit_elo_rating(self, epoch: int, miner_hotkey: str, rating: int, validator_hotkey: str = None, miner_response: str = None) -> bool:
        try:
            # Use stored validator hotkey if none provided
            if validator_hotkey is None and hasattr(self, 'validator_hotkey'):
                validator_hotkey = self.validator_hotkey
                bt.logging.debug(f"Using stored validator hotkey: {validator_hotkey}")
            
            if validator_hotkey is None:
                bt.logging.error("❌ No validator hotkey available for ELO rating submission")
                return False

            data = {
                'epoch': epoch,
                'miner_hotkey': miner_hotkey,
                'elo_rating': rating,
                'validator_hotkey': validator_hotkey,
                'miner_response': miner_response,
                'timestamp': time.time()
            }
            
            message = json.dumps(data, sort_keys=True, separators=(',', ':'))
            signature = self.validator_instance.wallet.hotkey.sign(message)


            payload = {
                "hotkey": self.validator_instance.wallet.hotkey.ss58_address,
                "data": data,
                "signature": signature.hex()
            }

            response = requests.post(self.api_url, json=payload, timeout=30)
            
            if response.status_code == 200 and response.json().get("success"):
                bt.logging.debug(f"ELO submitted: Epoch {epoch}, Miner {miner_hotkey[:10]}..., Rating {rating}")
                return True
            else:
                bt.logging.error(f"API error {response.status_code}: {response.text}")
                return False
            
        except Exception as e:
            bt.logging.error(f"❌ Failed to submit ELO rating: {e}")
            return False

    

    
    