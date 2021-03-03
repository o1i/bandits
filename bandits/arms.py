"""Valuation for the bandits.

This class can be used to calculate the arms of any arm given the input contexts.
Can be modified to add/remove arms or have non-stationary arms over time.

Currently the true expected arms follow multiples of gaussion mixtures on the context space onto which random
noise is added.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.spatial.distance import cdist


class Arm():
    """Superclass that represents an arm. See subclasses for details.
    Offers the following API:
    - value: get sample rewards
    - ev: get expected rewards (for arms)
    """
    def value(self, contexts: np.ndarray) -> tuple:
        """Realisations. NxD -> N, N returns observations and expected values (for regret)"""
        raise NotImplementedError

    def ev(self, context: np.ndarray) -> np.ndarray:
        """Expected value. NxD -> N"""
        raise NotImplementedError


class GaussianMixtureArm(Arm):
    """Expected value is a multiple of a sum of circular gaussian density functions on the context space"""
    def __init__(self, centres: np.array, stds: np.array, factor: float, noise: float):
        assert stds.shape[0] == centres.shape[0]
        self.centres = centres  # NCxD
        self.stds = stds        # NC
        self.factor = factor    # just a number
        self.noise = noise      # standard deviation of the noise

    def ev(self, contexts: np.ndarray):
        return (norm.pdf(cdist(contexts, self.centres) / np.tile(self.stds, [contexts.shape[0], 1]))
                * self.factor).sum(axis=1)

    def value(self, contexts: np.ndarray):
        ev = self.ev(contexts)
        return ev + np.random.normal(0, self.noise, ev.shape), ev
