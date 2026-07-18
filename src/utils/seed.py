import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG this codebase touches, including neat-python's.

    neat-python draws all of its randomness from the stdlib `random` module
    rather than a scoped RNG object, so seeding torch/numpy alone leaves
    evolution runs non-reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
