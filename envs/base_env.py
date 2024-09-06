import torch

# Base class/interface for ide
class BaseEnv:
    def __init__(self) -> None:
        pass
    def reset_env(self):
        pass
    def get_max_actions(self):
        pass
    def get_state_shape(self):
        return torch.rand(5)[0]
    def step(self, action: int):
        pass
    def compute_damage(self, potency: float):
        pass
    def state(self):
        return torch.rand(5)