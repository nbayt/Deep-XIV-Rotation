import torch

# Base class/interface for ide
class BaseEnv:
    
    def __init__(self) -> None:
        self.sks = 420
        pass
    def reset_env(self, _sks=None):
        """Resets the environment back to it's inital state, allows modification of the
        inital value for sks as needed."""
        pass
    def get_max_actions(self):
        """Returns the maximal actions for the environment."""
        pass
    def get_state_shape(self):
        """Returns a tensor with the number of features for this environment."""
        return torch.rand(5)[0]
    def valid_actions(self):
        return [True], [0]
    def step(self, action: int,  _verbose = False):
        """Attempts to perform the given action, stepping forward time and modifiying the interal state
        if necessary. Expects an Int for the action along with an optional boolean for verbose logging."""
        return (0.0, 1.0, 2.0)
    def compute_damage(self, potency: float):
        """Computes the damage of a given potency value, including variance, whether
        the hit was a crit or direct hit."""
        return 1.0
    def one_hot_encode(self, val, num_classes):
        ret = []
        for _ in range(num_classes):
            ret.append(0)
        ret[val] = 1
        return ret
    def state(self):
        """Returns a 1-dim Tensor containing the current values to represent the current state.\n
        These values may either be normalized or encoded in a different manner suitable for training."""
        return torch.rand(5)
    def is_done(self):
        """INOP - always returns False"""
        return False