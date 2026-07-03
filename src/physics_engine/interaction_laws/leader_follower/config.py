from dataclasses import dataclass


@dataclass
class LeaderFollowerConfig:
    """
    Parameters governing the leader-follower interaction model.
    """

    leader_id: int = 0

    leader_attraction_weight: float = 0.40
    leader_alignment_weight: float = 0.25
    separation_weight: float = 0.20
    wander_weight: float = 0.15

    leader_radius: float = 5.0
    separation_radius: float = 1.0

    wander_radius: float = 1.5
    wander_noise: float = 0.05

    max_separation_strength: float = 0.5