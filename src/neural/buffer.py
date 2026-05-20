"""
Reservoir buffer for Deep CFR experience replay.

Stores (features, regret_vector) pairs sampled uniformly from the
stream of visited information sets during traversal.
"""

import random
import numpy as np


class ReservoirBuffer:
    """Fixed-capacity buffer with reservoir sampling.

    When the buffer is full, each new item has a probability
    ``capacity / n_seen`` of replacing an existing item,
    ensuring a uniform sample over the stream seen so far.
    """

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.buffer = []          # list of (features, regrets)
        self.n_seen = 0

    def add(self, features, regrets):
        """Insert one training example."""
        if len(self.buffer) < self.capacity:
            self.buffer.append((features, regrets))
        else:
            # reservoir sampling: replace a random earlier item
            idx = random.randint(0, self.n_seen)
            if idx < self.capacity:
                self.buffer[idx] = (features, regrets)
        self.n_seen += 1

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        """Return a random mini-batch ``(features, targets)``."""
        n = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, n)
        feats = np.array([b[0] for b in batch], dtype=np.float32)
        tgts  = np.array([b[1] for b in batch], dtype=np.float32)
        return feats, tgts
