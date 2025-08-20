"""
Simple ELO Sync System
- No thresholds, no waiting
- Just collect all results and push to Bittensor
- Super simple but powerful
"""
import json
import time
from typing import Dict, List, Optional
import bittensor as bt
from supabase import create_client, Client


class SimpleELOSync:
    """
    Super simple ELO sync that just collects results and pushes to Bittensor.
    No thresholds, no waiting - just epoch → evaluate → collect → push.
    """
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.validator_instance = None
        
        # Test connection
        try:
            self.supabase.table('elo_submissions').select('id').limit(1).execute()
            bt.logging.info("✅ Simple ELO sync connected to Supabase")
        except Exception as e:
            bt.logging.error(f"❌ Failed to connect to Supabase: {e}")
            raise
    
    def set_validator_instance(self, validator):
        """Set validator instance for metagraph access"""
        self.validator_instance = validator
    
    def submit_elo_rating(self, epoch: int, miner_uid: int, rating: int, validator_uid: int) -> bool:
        """
        Submit ELO rating - no waiting, no thresholds.
        Just store the result.
        """
        try:
            result = self.supabase.table('elo_submissions').upsert({
                'epoch': epoch,
                'miner_uid': miner_uid,
                'elo_rating': rating,
                'validator_uid': validator_uid,
                'timestamp': time.time()
            }).execute()
            
            bt.logging.debug(f"✅ ELO rating submitted: Epoch {epoch}, Miner {miner_uid}, Rating {rating}")
            return True
            
        except Exception as e:
            bt.logging.error(f"❌ Failed to submit ELO rating: {e}")
            return False
    
    def finalize_epoch(self, epoch: int) -> Optional[Dict]:
        """
        Finalize epoch when it ends - collect ALL results.
        No thresholds, no waiting - just get everything.
        """
        try:
            # Get ALL submissions for this epoch
            result = self.supabase.table('elo_submissions').select('*').eq('epoch', epoch).execute()
            
            if not result.data:
                bt.logging.warning(f"⚠️ No submissions found for epoch {epoch}")
                return None
            
            # Group by miner and calculate consensus
            consensus = {}
            miner_ratings = {}
            
            # Group ratings by miner
            for submission in result.data:
                miner_uid = submission['miner_uid']
                rating = submission['elo_rating']
                
                if miner_uid not in miner_ratings:
                    miner_ratings[miner_uid] = []
                miner_ratings[miner_uid].append(rating)
            
            # Calculate consensus for each miner
            for miner_uid, ratings in miner_ratings.items():
                if not ratings:
                    continue
                
                # Simple average - no stake weighting
                avg_elo = sum(ratings) / len(ratings)
                weight = min(1.0, avg_elo / 1000.0)  # Normalize to 0-1
                
                consensus[miner_uid] = {
                    'final_elo': int(avg_elo),
                    'final_weight': weight,
                    'validator_count': len(ratings),
                    'ratings': ratings
                }
            
            # Store consensus
            consensus_result = self.supabase.table('epoch_consensus').upsert({
                'epoch': epoch,
                'consensus_data': consensus,
                'finalized': True,
                'finalized_at': time.time(),
                'pushed_to_bittensor': False
            }).execute()
            
            bt.logging.info(f"🎯 Epoch {epoch} finalized: {len(consensus)} miners, {len(result.data)} total ratings")
            return consensus
            
        except Exception as e:
            bt.logging.error(f"❌ Failed to finalize epoch {epoch}: {e}")
            return None
    
    def push_to_bittensor(self, epoch: int) -> bool:
        """
        Push consensus weights to Bittensor.
        Called before next epoch starts.
        """
        try:
            # Get consensus for this epoch
            result = self.supabase.table('epoch_consensus').select('*').eq('epoch', epoch).eq('finalized', True).execute()
            
            if not result.data:
                bt.logging.warning(f"⚠️ No consensus found for epoch {epoch}")
                return False
            
            consensus_data = result.data[0]['consensus_data']
            
            if not self.validator_instance:
                bt.logging.error("❌ No validator instance set")
                return False
            
            # Create weight tensor
            metagraph = self.validator_instance.metagraph
            weights = {}
            
            for miner_uid, data in consensus_data.items():
                uid = int(miner_uid)
                if uid < len(metagraph.hotkeys):
                    weights[uid] = data['final_weight']
                    bt.logging.debug(f"⚖️ Miner {uid}: Weight {data['final_weight']:.4f}")
            
            # Set weights on Bittensor
            if weights:
                self.validator_instance.set_weights_from_consensus(weights)
                
                # Mark as pushed
                self.supabase.table('epoch_consensus').update({
                    'pushed_to_bittensor': True
                }).eq('epoch', epoch).execute()
                
                bt.logging.info(f"🚀 Epoch {epoch} weights pushed to Bittensor: {len(weights)} miners")
                return True
            else:
                bt.logging.warning(f"⚠️ No valid weights to push for epoch {epoch}")
                return False
                
        except Exception as e:
            bt.logging.error(f"❌ Failed to push epoch {epoch} to Bittensor: {e}")
            return False
    
    def get_epoch_status(self, epoch: int) -> Dict:
        """Get simple status for an epoch"""
        try:
            # Get submission count
            submissions = self.supabase.table('elo_submissions').select('validator_uid').eq('epoch', epoch).execute()
            unique_validators = len(set(sub['validator_uid'] for sub in submissions.data)) if submissions.data else 0
            
            # Get consensus status
            consensus = self.supabase.table('epoch_consensus').select('*').eq('epoch', epoch).execute()
            
            status = {
                'epoch': epoch,
                'validators_participated': unique_validators,
                'total_submissions': len(submissions.data) if submissions.data else 0,
                'finalized': False,
                'pushed_to_bittensor': False
            }
            
            if consensus.data:
                status['finalized'] = consensus.data[0]['finalized']
                status['pushed_to_bittensor'] = consensus.data[0]['pushed_to_bittensor']
            
            return status
            
        except Exception as e:
            bt.logging.error(f"❌ Failed to get epoch {epoch} status: {e}")
            return {'epoch': epoch, 'error': str(e)}
    
    def cleanup_old_epochs(self, keep_last: int = 10):
        """Keep only recent epochs"""
        try:
            # Get all epochs
            result = self.supabase.table('epoch_consensus').select('epoch').order('epoch', desc=True).execute()
            
            if not result.data:
                return
            
            epochs = [row['epoch'] for row in result.data]
            if len(epochs) <= keep_last:
                return
            
            # Delete old epochs
            cutoff_epoch = epochs[keep_last]
            
            # Delete old submissions
            self.supabase.table('elo_submissions').delete().lt('epoch', cutoff_epoch).execute()
            
            # Delete old consensus
            self.supabase.table('epoch_consensus').delete().lt('epoch', cutoff_epoch).execute()
            
            bt.logging.info(f"🗑️ Cleaned up epochs older than {cutoff_epoch}")
            
        except Exception as e:
            bt.logging.error(f"❌ Failed to cleanup old epochs: {e}")
