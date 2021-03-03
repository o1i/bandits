"""
This module handles the contexts to be used for the contextual bandits.
The goal is to allow for time-varying contexts where the variation can either be gradual or sudden.
The regrets are (random) functions in the context space.

Overall the assumption to date is that the cnotexts are centred multivariate normal.

The assumption is that the learners are myopic, i.e. that they do not track individual contexts over multiple time
periods. This can be justified in cases where there are lots of observed contexts but learning needs to happen
in a time frame that is a lot shorter than the time life span of the entity that is described by a context.
"""
import numpy as np


class Context:
    def __init__(self, n: int, d: int):
        """Creates the inital context for the bandits"""
        self.contexts = np.random.normal(0, 1, (n, d))

    def step(self, shrink: float = 0.05, move: float = 0.1):
        """Random walk with drift to 0"""
        self.contexts = self.contexts * (1-shrink) + move * np.random.normal(0, 1, self.contexts.shape)
