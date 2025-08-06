import torch
import random
import bittensor as bt
from typing import List


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


def round_robin_sample(validator, candidate_uids: List[int], k: int) -> List[int]:
    """
    Round-robin sampling to ensure fair evaluation of all miners.
    Each miner gets evaluated exactly once per complete cycle.
    
    Args:
        validator: Validator instance containing global_miner_index
        candidate_uids (List[int]): List of available miner UIDs
        k (int): Number of miners to select
        
    Returns:
        List[int]: Selected miner UIDs in round-robin order
    """
    if not candidate_uids:
        return []
    
    if len(candidate_uids) <= k:
        # If we have fewer candidates than requested, return all
        return candidate_uids
    
    # Round-robin selection
    selected = []
    total_candidates = len(candidate_uids)
    
    # Get current position and wrap around if needed
    start_index = validator.global_miner_index % total_candidates
    
    for i in range(k):
        # Select miners in round-robin fashion
        index = (start_index + i) % total_candidates
        selected.append(candidate_uids[index])
    
    # Update global index for next round
    validator.global_miner_index = (validator.global_miner_index + k) % total_candidates
    
    bt.logging.debug(f"Round-robin: selected {len(selected)} miners starting from index {start_index}")
    
    return selected





def get_miners_uids(self, k: int, exclude: List[int] = None, specified_miners: List[int] = None) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    if specified_miners is not None:
        specified_miners_set = set(specified_miners)  # Use set for faster lookup
    else:
        specified_miners_set = None

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

    k = min(k, len(candidate_uids))
    uids = torch.tensor(round_robin_sample(self, candidate_uids, k))
    return uids


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
