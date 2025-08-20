from pydantic import BaseModel, Field
from typing import List, Dict, Any


class Rubric(BaseModel):
    """Rubric for a criterion."""
    
    rubric_id: str = Field(description="Unique identifier for the criterion")
    mark: int = Field(description="Mark for the criterion")
    

class MinerEvaluation(BaseModel):
    """Individual miner evaluation for a tournament round."""
    
    miner_id: int = Field(description="Unique identifier for the miner")
    criteria_scores: List[Rubric] = Field(description="Scores for each miner")
    

class TournamentResult(BaseModel):
    """Complete tournament round result with all miner evaluations."""
    
    miner_evaluations: List[MinerEvaluation] = Field(
        description="Evaluations for all miners in this round",
        min_items=1
    )
    