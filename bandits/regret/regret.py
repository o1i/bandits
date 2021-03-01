"""Valuation for the bandits.

This class can be used to calculate the regret of any arm given the input contexts.
Can be modified to add/remove arms or have non-stationary regret over time.

Currently the true expected regret follow multiples of gaussion mixtures on the context space onto which random
noise is added.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm


class Regrets:
    def __init__(self, n_arms: int, n_centres: int, d: int, sigma_mu: float, sigma: tuple, r: tuple):
        """
        Initialsises regret setup.
        For a given arm A the expected value of an arm A is given as
        V_A := r_A * sum_{i=0}^{n_centres} phi((c-mu_i)^T (c_mu_i) / sigma_i)
        where phi denotes the density of the standard normal
        The regret of arm A is defined as R_A := -(V_A - max_Ai(V_Ai).)

        :param n_arms: number of arms
        :param n_centres: number of centres per arm
        :param d: dimensionality of the context space
        :param sigma_mu: standard deviation for the value centres
        :param sigma: lower and upper bound of the uniform distribution setting the std of the value centres
        :param r: lower und upper bound of the uniform distribution setting the value multiplier
        """
        self.n_arms = n_arms
        self.n_centres = n_centres
        self.d = d
        self.sigma_mu = sigma_mu
        self.sigma = sigma
        self.r = r
        self.arms = defaultdict(dict)
        for i in range(n_arms):
            self.arms[i]["mu"] = np.random.normal(0, sigma_mu, (d, n_centres))
            self.arms[i]["sigma"] = np.random.uniform(sigma[0], sigma[1], n_centres)
            self.arms[i]["r"] = np.random.uniform(r[0], r[1])

    def regret(self, context: np.ndarray) -> pd.DataFrame:
        """
        Given a context of size NxD, returns the regret in the shape of NxN_A.
        The context must match with the dimension chosen in the initialization of the Regrets class.
        The return value is a dataframe to be able to identify the arms in the case of added/deleted arms
        :param context: matrix with the contexts
        :return: Non-negative dataframe of the regret that contains one zero value on every row for the optimal choice.
        """
        assert context.shape[1] == self.d, "Incompatible context and Regret instance"
        values = np.ndarray(shape=(context.shape[0], self.n_arms))
        arm_names = []
        n = context.shape[0]
        for i, (arm_name, spec) in enumerate(self.arms.items()):
            values[:, i] = norm.pdf(
                ((np.stack([context] * self.n_centres, -1) -
                  np.stack([spec["mu"]]*n, 0)) ** 2
                 ).sum(axis=1) / np.stack(spec["sigma"] * n, 0)
            ).sum(axis=1) * spec["r"]
            arm_names.append(arm_name)
        regret = - (values - np.stack([values.max(axis=1)] * values.shape[1], -1))
        return pd.DataFrame(regret, columns = arm_names)
