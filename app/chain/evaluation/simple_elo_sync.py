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
from supabase import create_client, Client


class EloManager:
    """
    Independent ELO sync - each validator works independently.
    No consensus, no waiting - just store scores and push weights.
    """
    
    def __init__(self, supabase_url: str, supabase_key: str, validator_instance=None):
        # Test connection BEFORE creating the object
        test_client = create_client(supabase_url, supabase_key)
        try:
            # Create a test record to verify full database functionality
            test_data = {
                'epoch': -1,  # Use negative epoch to avoid conflicts
                'miner_hotkey': 'test_miner',
                'elo_rating': 1000,
                'validator_hotkey': 'test_validator',
                'timestamp': time.time()
            }
            
            # Insert test data
            insert_result = test_client.table('elo_submissions').insert(test_data).execute()
            bt.logging.info("✅ Test data inserted successfully")
            
            # Verify we can read it back
            read_result = test_client.table('elo_submissions').select('*').eq('epoch', -1).execute()
            if read_result.data:
                bt.logging.info("✅ Test data read successfully")
            else:
                raise Exception("Could not read test data back")
            
            # Clean up test data
            delete_result = test_client.table('elo_submissions').delete().eq('epoch', -1).execute()
            bt.logging.info("✅ Test data cleaned up successfully")
            
            bt.logging.info("✅ Simple ELO sync connected to Supabase with full CRUD functionality")
            
        except Exception as e:
            bt.logging.error(f"❌ Failed to connect to Supabase: {e}")
            raise
        
        # Only create the object if connection test passes
        self.supabase: Client = test_client
        self.validator_instance = validator_instance
        self.validator_hotkey = None
    
    def set_validator_hotkey(self, validator_hotkey: str):
        """Set validator hotkey for ELO sync operations"""
        self.validator_hotkey = validator_hotkey
        bt.logging.info(f"✅ Validator hotkey set to {validator_hotkey} in ELO sync manager")
    
    def submit_elo_rating(self, epoch: int, miner_hotkey: str, rating: int, validator_hotkey: str = None) -> bool:
        try:
            # Use stored validator hotkey if none provided
            if validator_hotkey is None and hasattr(self, 'validator_hotkey'):
                validator_hotkey = self.validator_hotkey
                bt.logging.debug(f"Using stored validator hotkey: {validator_hotkey}")
            
            if validator_hotkey is None:
                bt.logging.error("❌ No validator hotkey available for ELO rating submission")
                return False
            
            result = self.supabase.table('elo_submissions').upsert({
                'epoch': epoch,
                'miner_hotkey': miner_hotkey,
                'elo_rating': rating,
                'validator_hotkey': validator_hotkey,
                'timestamp': time.time()
            }).execute()
            
            bt.logging.debug(f"✅ ELO rating submitted: Epoch {epoch}, Miner {miner_hotkey}, Rating {rating}, Validator {validator_hotkey}")
            return True
            
        except Exception as e:
            bt.logging.error(f"❌ Failed to submit ELO rating: {e}")
            return False
    

    
    def push_weights_to_bittensor(self, epoch: int, validator_hotkey: str) -> bool:
        """
        Push validator's own weights to Bittensor.
        Each validator pushes their own weights independently.
        """
        try:
            # Get final scores for this validator in this epoch
            result = self.supabase.table('validator_final_scores').select('*').eq('epoch', epoch).eq('validator_hotkey', validator_hotkey).execute()
            
            if not result.data:
                bt.logging.warning(f"⚠️ No final scores found for validator {validator_hotkey} in epoch {epoch}")
                return False
            
            if not self.validator_instance:
                bt.logging.error("❌ No validator instance set")
                return False
            
            # Create weight tensor from this validator's scores
            metagraph = self.validator_instance.metagraph
            weights = {}
            
            for score_data in result.data:
                miner_hotkey = score_data['miner_hotkey']
                final_weight = score_data['final_weight']
                
                # Find UID for this hotkey in metagraph
                uid = None
                for i, hotkey in enumerate(metagraph.hotkeys):
                    if hotkey == miner_hotkey:
                        uid = i
                        break
                
                if uid is not None and uid < len(metagraph.hotkeys):
                    weights[uid] = final_weight
                    bt.logging.debug(f"⚖️ Miner {miner_hotkey} (UID {uid}): Weight {final_weight:.4f}")
                else:
                    bt.logging.warning(f"⚠️ Miner hotkey {miner_hotkey} not found in metagraph")
            
            # Set weights on Bittensor
            if weights:
                self.validator_instance.set_weights_from_consensus(weights)
                bt.logging.info(f"🚀 Epoch {epoch} weights pushed to Bittensor by validator {validator_hotkey}: {len(weights)} miners")
                return True
            else:
                bt.logging.warning(f"⚠️ No valid weights to push for epoch {epoch}")
                return False
                
        except Exception as e:
            bt.logging.error(f"❌ Failed to push epoch {epoch} to Bittensor: {e}")
            return False
