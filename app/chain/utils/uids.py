import torch
import random
import bittensor as bt
from typing import List
import json
import random
import os
import hashlib
from typing import List, Tuple
from dotenv import load_dotenv


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    
    # Note: We no longer filter by IP address here since we want to include
    # non-working miners but give them zero scores after normalization
    
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def create_balanced_groups(candidate_uids: List[int], target_size: int = 6, random_seed: int = None) -> List[List[int]]:
    """
    Creates balanced groups from a list of candidate UIDs.
    
    Args:
        candidate_uids (List[int]): List of candidate UIDs to group
        target_size (int): Target number of miners per group (default: 6)
        random_seed (int): Optional random seed for deterministic shuffling
        
    Returns:
        List[List[int]]: List of balanced groups
    """
    if not candidate_uids:
        return []

    total_candidates = len(candidate_uids)
    base_groups = total_candidates // target_size
    remainder = total_candidates % target_size

    groups = []
    shuffled_uids = candidate_uids.copy()

    # Shuffle with provided or random seed
    if random_seed is None:
        random_seed = random.randint(1, 1000000)
    
    random.seed(random_seed)
    random.shuffle(shuffled_uids)

    if remainder == 0:
        # Perfect division: all groups of target size
        for i in range(base_groups):
            start_idx = i * target_size
            end_idx = start_idx + target_size
            groups.append(shuffled_uids[start_idx:end_idx])
            
    elif remainder == 1:
        # For remainder 1, create balanced groups
        miners_used = (base_groups - 1) * target_size + 1 * (target_size - 1) + 2 * (target_size - 1)
        
        if miners_used <= total_candidates:
            # Create (base_groups - 1) groups of target_size
            for i in range(base_groups - 1):
                start_idx = i * target_size
                end_idx = start_idx + target_size
                groups.append(shuffled_uids[start_idx:end_idx])
            
            # Last group becomes smaller
            last_start = (base_groups - 1) * target_size
            groups.append(shuffled_uids[last_start:last_start + (target_size - 1)])
            
            # Create 2 groups with remaining miners
            remaining_start = last_start + (target_size - 1)
            groups.append(shuffled_uids[remaining_start:remaining_start + (target_size - 1)])
            groups.append(shuffled_uids[remaining_start + (target_size - 1):remaining_start + 2 * (target_size - 1)])
        else:
            # Fallback: create balanced groups without exceeding available miners
            for i in range(base_groups - 1):
                start_idx = i * target_size
                end_idx = start_idx + target_size
                groups.append(shuffled_uids[start_idx:end_idx])
            
            # Last group becomes smaller
            last_start = (base_groups - 1) * target_size
            groups.append(shuffled_uids[last_start:last_start + (target_size - 1)])
            
            # Create 1 group with remaining miners
            remaining_start = last_start + (target_size - 1)
            remaining_count = total_candidates - remaining_start
            if remaining_count > 0:
                groups.append(shuffled_uids[remaining_start:remaining_start + remaining_count])
        
    elif remainder == 2:
        # For remainder 2, create balanced groups
        miners_used = (base_groups - 2) * target_size + 2 * (target_size - 1) + 2 * (target_size + 1)
        
        if miners_used <= total_candidates:
            # Create (base_groups - 2) groups of target_size
            for i in range(base_groups - 2):
                start_idx = i * target_size
                end_idx = start_idx + target_size
                groups.append(shuffled_uids[start_idx:end_idx])
            
            # Last 2 groups become smaller
            last_2_start = (base_groups - 2) * target_size
            groups.append(shuffled_uids[last_2_start:last_2_start + (target_size - 1)])
            groups.append(shuffled_uids[last_2_start + (target_size - 1):last_2_start + 2 * (target_size - 1)])
            
            # Create 2 larger groups with remaining miners
            remaining_start = last_2_start + 2 * (target_size - 1)
            groups.append(shuffled_uids[remaining_start:remaining_start + (target_size + 1)])
            groups.append(shuffled_uids[remaining_start + (target_size + 1):remaining_start + 2 * (target_size + 1)])
        else:
            # Fallback: create balanced groups without exceeding available miners
            for i in range(base_groups - 1):
                start_idx = i * target_size
                end_idx = start_idx + target_size
                groups.append(shuffled_uids[start_idx:end_idx])
            
            # Last group becomes smaller
            last_start = (base_groups - 1) * target_size
            groups.append(shuffled_uids[last_start:last_start + (target_size - 1)])
            
            # Create 1 group with remaining miners
            remaining_start = last_start + (target_size - 1)
            remaining_count = total_candidates - remaining_start
            if remaining_count > 0:
                groups.append(shuffled_uids[remaining_start:remaining_start + remaining_count])
        
    elif remainder == 3:
        # For remainder 3, create balanced groups
        miners_used = (base_groups - 2) * target_size + 2 * (target_size - 1) + 1 * (target_size + 1) + 1 * (target_size - 1)
        
        if miners_used <= total_candidates:
            # Create (base_groups - 2) groups of target_size
            for i in range(base_groups - 2):
                start_idx = i * target_size
                end_idx = start_idx + target_size
                groups.append(shuffled_uids[start_idx:end_idx])
            
            # Last 2 groups become smaller
            last_2_start = (base_groups - 2) * target_size
            groups.append(shuffled_uids[last_2_start:last_2_start + (target_size - 1)])
            groups.append(shuffled_uids[last_2_start + (target_size - 1):last_2_start + 2 * (target_size - 1)])
            
            # Create 1 larger group + 1 smaller group with remaining miners
            remaining_start = last_2_start + 2 * (target_size - 1)
            groups.append(shuffled_uids[remaining_start:remaining_start + (target_size + 1)])
            groups.append(shuffled_uids[remaining_start + (target_size + 1):remaining_start + (target_size + 1) + (target_size - 1)])
        else:
            # Fallback: create balanced groups without exceeding available miners
            for i in range(base_groups - 1):
                start_idx = i * target_size
                end_idx = start_idx + target_size
                groups.append(shuffled_uids[start_idx:end_idx])
            
            # Last group becomes smaller
            last_start = (base_groups - 1) * target_size
            groups.append(shuffled_uids[last_start:last_start + (target_size - 1)])
            
            # Create 1 group with remaining miners
            remaining_start = last_start + (target_size - 1)
            remaining_count = total_candidates - remaining_start
            if remaining_count > 0:
                groups.append(shuffled_uids[remaining_start:remaining_start + remaining_count])
        
    elif remainder == 4:
        # Create base_groups groups of target_size, 1 group of 4
        for i in range(base_groups):
            start_idx = i * target_size
            end_idx = start_idx + target_size
            groups.append(shuffled_uids[start_idx:end_idx])
        
        # Create 1 group of 4 with remaining miners
        remaining_start = base_groups * target_size
        groups.append(shuffled_uids[remaining_start:remaining_start + 4])
        
    elif remainder == 5:
        # Create base_groups groups of target_size, 1 group of 5
        for i in range(base_groups):
            start_idx = i * target_size
            end_idx = start_idx + target_size
            groups.append(shuffled_uids[start_idx:end_idx])
        
        # Create 1 group of 5 with remaining miners
        remaining_start = base_groups * target_size
        groups.append(shuffled_uids[remaining_start:remaining_start + 5])

    return groups


def tournament_group_miners(self, exclude: List[int] = None, specified_miners: List[int] = None, target_size: int = 6) -> List[List[int]]:
    """
    Creates tournament groups of miners using optimal grouping (4-7 miners per group).
    
    Args:
        exclude (List[int]): List of UIDs to exclude from selection
        specified_miners (List[int]): List of specific miners to use
        target_size (int): Target number of miners per group (default: 6)
        
    Returns:
        List[List[int]]: List of tournament groups
    """
    candidate_uids = []
    avail_uids = []

    if specified_miners is not None:
        specified_miners_set = set(specified_miners)
    else:
        specified_miners_set = None

    # Collect candidate UIDs based on availability and filters
    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude
        uid_is_in_specified = specified_miners_set is None or uid in specified_miners_set

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded and uid_is_in_specified:
                candidate_uids.append(uid)

    if not candidate_uids:
        return []

    # Generate deterministic seed for grouping consistency
    current_epoch = getattr(self, 'current_epoch', 0)
    evaluation_round = getattr(self, 'evaluation_round', 0)
    random_seed = random.randint(1, 1000000)  # Random seed for each evaluation cycle
    
    seed = f"{current_epoch}_{evaluation_round}_{random_seed}_{len(candidate_uids)}"
    hash_seed = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)

    # Use the separated grouping function
    return create_balanced_groups(candidate_uids, target_size, hash_seed)

def get_miners_uids(self, k: int, exclude: List[int] = None, specified_miners: List[int] = None) -> torch.LongTensor:
    """
    Returns miners for tournament evaluation using the new grouping system.
    
    Args:
        k (int): Number of miners to return (will be adjusted to group size)
        exclude (List[int]): List of UIDs to exclude from selection
        specified_miners (List[int]): List of specific miners to use
        
    Returns:
        uids (torch.LongTensor): Miners selected for tournament evaluation
    """
    # Default group size, can be made configurable
    group_size = getattr(self, 'tournament_group_size', 6)
    
    # Get tournament groups
    groups = self.tournament_group_miners(exclude, specified_miners, group_size)
    
    if not groups:
        return torch.tensor([])
    
    # Select first group for evaluation (can be modified to select random group)
    selected_group = groups[0]
    
    bt.logging.info(f"Selected tournament group: {selected_group}")
    return torch.tensor(selected_group)


def get_validator_uids(self, remove_self: bool = True) -> torch.LongTensor:
    """
    Returns all validator UIDs from the given metagraph, i.e. all UIDs
    where metagraph.validator_permit[uid] == True.
    """

    bt.logging.info(f"Getting validator uids from metagraph.")
    bt.logging.info(f"Validator permit: {self.metagraph.validator_permit}")
    validator_permit = torch.tensor(self.metagraph.validator_permit, dtype=torch.bool)
    validator_uids = torch.where(validator_permit)[0].long()
    if remove_self:
        self_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Self UID: {self_uid}")
        validator_uids = validator_uids[validator_uids != self_uid]

    bt.logging.info(f"Validator UIDs (after removal): {validator_uids}")
    return validator_uids
