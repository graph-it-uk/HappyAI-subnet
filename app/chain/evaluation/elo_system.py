import json
import random
from typing import List, Dict, Tuple, Optional
import bittensor as bt


class ELOSystem:
    """
    ELO rating system for tournament-based miner evaluation.
    Implements Swiss Tournament + ELO hybrid approach.
    """
    
    def __init__(self, initial_rating: int = 1000, k_factor: int = 32, sync_manager: Optional[object] = None):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: Dict[int, int] = {}
        self.tournament_history: List[Dict] = []
        self.epoch_count = 0
        self.evaluation_round = 0  # Track evaluation rounds within epoch
        
        # ELO synchronization manager
        self.sync_manager = sync_manager
        self.use_sync = sync_manager is not None
        
    def initialize_miner(self, uid: int):
        """Initialize a new miner with default ELO rating."""
        if uid not in self.ratings:
            self.ratings[uid] = self.initial_rating
            
    def get_rating(self, uid: int) -> int:
        """Get current ELO rating for a miner."""
        if uid not in self.ratings:
            self.initialize_miner(uid)
        return self.ratings[uid]
                

        
    def get_rankings(self) -> List[Tuple[int, int]]:
        """Get sorted list of (uid, rating) tuples."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        
    def reset_ratings(self):
        """Reset all ELO ratings to initial value (called every 10 epochs)."""
        for uid in self.ratings:
            self.ratings[uid] = self.initial_rating
        bt.logging.info("ELO ratings reset to initial values")
        
    def increment_epoch(self):
        """Increment epoch counter and reset ratings if needed."""
        self.epoch_count += 1
        self.evaluation_round = 0  # Reset evaluation round counter
        if self.epoch_count % 10 == 0:  # Reset every 10 epochs
            self.reset_ratings()
            
    def increment_evaluation_round(self):
        """Increment evaluation round counter within epoch."""
        self.evaluation_round += 1
    


