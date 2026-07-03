from dataclasses import dataclass

@dataclass
class Spring:
    """
    Generic agent used by every interaction law.
    The force acts on the agent.
    The spring is just a relationship.
    """
    agent_a: int
    agent_b: int

    rest_length: float
    stiffness: float