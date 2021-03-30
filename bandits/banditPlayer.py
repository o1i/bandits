from collections import OrderedDict

import numpy as np


class BanditPlayer:
    """
        Bandit to be used in learning. Offers the following API:

        - __init__(arms: dict)              # set up bandit with a dict of named arms
        - options() -> list[str]            # returns the names of the currently available arms
        - play_all(context, choices) -> np.array, np.array   # given contexts and a list of played arm-names for
                                                               each context, return value and regret
        - play_one(context, choice) -> float, float  # given a single context gets observed value and regret
    """
    def __init__(self, arms: dict):
        """
        :param arms: Arms to be used at initialisation
        """
        self.arms = OrderedDict(arms)

    def options(self):
        return list(self.arms.keys())

    def play_one(self, context: np.array, choice: str):
        """
        Given a context and a choice of action, gets the observed value for the chosen action and the regret
        (max(ev_{choices}) - ev[choice]) of that action.
        :param context: 1-dim array in the feature space of the arms
        :param choice: name of the arm
        :return: observed_value, regret_of_choice
        """
        assert choice in self.arms.keys()
        arms = list(self.arms.keys())
        all_obs = np.zeros(len(arms))
        all_ev = np.zeros(len(arms))
        for i, arm in enumerate(arms):
            obs, ev = self.arms[arm].value(context.reshape([1, -1]))
            all_obs[i] = obs
            all_ev[i] = ev
        ind_choice = arms.index(choice)
        return all_obs[ind_choice], max(all_ev) - all_ev[ind_choice]

    def play_all(self, contexts: np.array, choices: list):
        """
        Given multiple contexts and corresponding choices, returns observed values and regrets
        :param contexts: NxD, N contexts in D dimensional arm-feature-space
        :param choices: N, names of the chosen arms
        :return: N, N the observed values and the (unobserved) regrets
        """
        assert contexts.shape[0] == len(choices), "All contexts must have a choice"
        n, d = contexts.shape
        arms, obs, ev = self._get_values(contexts)
        regrets = ev.max(axis=1).tile([1, d]) - ev
        assert not set(choices) - set(arms), "Invalid choice"
        indices = [arms.index(c) for c in choices]
        all_inds = np.arange(n)
        return obs[all_inds, indices], regrets[all_inds, indices]

    def _get_values(self, contexts: np.array):
        """
        Gets the list of arms, observed values and unobserved expected values as arrays
        :param contexts: NxD
        :return: list of arm names, array of observed values, array of (unobserved) expected values
        """
        arms = list(self.arms.keys())
        obs = ev = np.zeros([contexts.shape[0], len(arms)])
        for i, arm in enumerate(arms):
            observed, expected = self.arms[arm].value(contexts)
            obs[:, i] = observed
            ev[:, i] = expected
        return arms, obs, ev
