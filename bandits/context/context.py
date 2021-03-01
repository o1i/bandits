import numpy as np


class Context:
    def __init__(self, n: int, d: int):
        """Creates the inital context for the bandits"""
        self.contexts = np.random.normal(0, 1, (n, d))

    def step(self, shrink: float = 0.05, move: float = 0.1):
        self.contexts = self.contexts * (1-shrink) + move * np.random.normal(0, 1, self.contexts.shape)
