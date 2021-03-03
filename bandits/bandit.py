import numpy as np

from bandits.arms import Arm


class Bandit:
    """
        Bandit to be used in learning. Offers the following API:

        - __init__(arms: dict)              # set up bandit with a dict of named arms
        - options() -> list[str]            # returns the names of the currently available arms
        - play(context, choices) -> np.array, np.array   # given contexts and a list of played arm-names for
                                                           each context, return value and regret
    """
    def __init__(self, arms: dict[str, Arm]):
        """
        :param arms: Arms to be used at initialisation
        """
        self.arms = arms

    def options(self):
        return list(self.arms.keys())

    def play(self, contexts: np.array, choices: list):
        assert contexts.shape[0] == len(choices), "All contexts must have a choice"
        n, d = contexts.shape
        arms, obs, ev = self._get_values(contexts)
        regrets = ev.max(axis=1).tile([1, d]) - ev
        assert not set(choices) - set(arms), "Invalid choice"
        indices = [arms.index(c) for c in choices]
        all_inds = np.arange(n)
        return obs[all_inds, indices], regrets[all_inds, indices]

    def _get_values(self, contexts: np.array):
        """Gets the values of all arms and returns observations and (true) expectations in two arrays"""
        arms = list(self.arms.keys())
        obs = ev = np.zeros([contexts.shape[0], len(arms)])
        for i, arm in enumerate(arms):
            observed, expected = self.arms[arm].value(contexts)
            obs[:, i] = observed
            ev[:, i] = expected
        return arms, obs, ev
