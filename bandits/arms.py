"""Ground truth value models for the bandits

An arm is a model R^D -> R that takes a D-dimensional input and returns a value composed of a deterministic part and
a random part with 0 expected value.

Currently the true expected values follow multiples of gaussion mixtures on the context space onto which random normal
noise is added.
"""
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

    def move_centres(self, std=1, abs_diff: np.ndarray = None):
        """Shifts centres either randomly or by absolute numbers. In the latter case, array dimension must match"""
        if abs_diff is not None:
            assert self.centres.shape == abs_diff.shape, f"Shape was {abs_diff.shape}, should be {self.centres.shape}"
            self.centres += abs_diff
        else:
            self.centres = self.centres + np.random.normal(0, std, self.centres.shape)
