"""Bandit learners ar the entities that learn which arms to play in which contexts

Here is where the ML happens.
"""
from collections import defaultdict
import numpy as np
from sklearn.linear_model import SGDRegressor
import xgboost as xgb

import river


class BanditLearner():
    """Class that has models for every arm and makes decsions based on the best outcome

    Offers the following API:
    - choose(s: np.array) -> str   # Chooses the arm to pick for input states (contexts)
    - update(s, a, r) -> None      # Updates the internal model(s) with state and reward
    """
    def __init__(self, learners: dict):
        self.learners = learners

    def choose(self, s: np.array):
        values = {k: v.predict(s) for k, v in self.learners.items()}
        return max(values, key=values.get)

    def update(self, s, a, r):
        raise NotImplementedError


class SGDLearner(BanditLearner):
    """Uses SGDRegressor as learners"""
    def __init__(self, n, loss="huber", penalty="l2", learning_rate="constant", eta0=0.01):
        learners = dict()
        for i in range(n):
            learners[f"a{i}"] = SGDRegressor(loss=loss, penalty=penalty, learning_rate=learning_rate, eta0=eta0)
        super().__init__(learners)

    def update(self, s, a, r):
        self.learners[a].partial_fit(s, np.array(r))


class XGBLearner(BanditLearner):
    """Uses XGBoost as learner"""
    def __init__(self, n_learners: int, n_trees: int, n_retrain=100, n_max=10000):
        learners = dict()
        self.obs = dict()  # to hold 4-tuples of states, rewards and "counts since last training"
        self.n_retrain = n_retrain
        self.n_max = n_max
        for i in range(n_learners):
            learners[f"a{i}"] = xgb.XGBRegressor(n_estimators=n_trees)
        super().__init__(learners)

    def update(self, s, a, r):
        if not a in self.obs.keys():
            self.obs[a] = (s, r, 0)
            self.learners[a].fit(s, r)
        else:
            s_old, r_old, count_old = self.obs[a]
            s_new = np.concatenate([s_old, s], 0)[-self.n_max:, :]
            r_new = np.concatenate([r_old, r])[-self.n_max:]
            count_new = count_old + s.shape[0]
            if count_new > self.n_retrain:
                self.learners[a].fit(s_new, r_new)
                count_new = 0
            self.obs[a] = (s_new,
                           r_new,
                           count_new)


class OptimisticSGDLearner(SGDLearner):
    """Uses SGDRegressor as learners"""
    def __init__(self, *args, alpha: float=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_error = defaultdict(lambda: 0)
        self.alpha = alpha

    def update(self, s, a, r):
        try:
            pred = self.learners[a].predict(s)
        except:
            pred = 0
        self.pred_error[a] = self.alpha * self.pred_error[a] + (1 - self.alpha) * sum(abs(pred - np.array(r)))
        self.learners[a].partial_fit(s, np.array(r))

    def choose(self, s: np.array):
        values = {k: v.predict(s) + (self.pred_error[k] if self.pred_error[k] else 0) for k, v in self.learners.items()}
        return max(values, key=values.get)


class AdaptiveRandomForestLearner(BanditLearner):
    def __init__(self, n_learners: int, n_trees: int = 21, eps=0.1, **kwargs):
        learners = dict()
        for i in range(n_learners):
            learners[f"a{i}"] = river.ensemble.AdaptiveRandomForestRegressor(n_models=n_trees, **kwargs)
        self.eps = eps
        super().__init__(learners)

    def update(self, s, a, r):
        self.learners[a].learn_one({str(i): v for i, v in enumerate(s.squeeze())}, r)

    def choose(self, s: np.array):
        if np.random.random() > self.eps:
            values = {k: v.predict_one({str(i): v for i, v in enumerate(s.squeeze())}) for k, v in self.learners.items()}
            return max(values, key=values.get)
        return np.random.choice(list(self.learners.keys()))


class PerceptronLearner(BanditLearner):  # Useless, this is a classifier
    def __init__(self, n_learners: int, l2: int = 0, eps=0.1):
        learners = dict()
        for i in range(n_learners):
            learners[f"a{i}"] = river.preprocessing.StandardScaler() | river.linear_model.Perceptron(l2=l2)
        self.eps = eps
        super().__init__(learners)

    def update(self, s, a, r):
        self.learners[a].learn_one({str(i): v for i, v in enumerate(s.squeeze())}, r)

    def choose(self, s: np.array):
        if np.random.random() > self.eps:
            values = {k: v.predict_one({str(i): v for i, v in enumerate(s.squeeze())}) for k, v in self.learners.items()}
            return max(values, key=values.get)
        return np.random.choice(list(self.learners.keys()))


class BaggedLinearRegressor(BanditLearner):
    def __init__(self, n_learners: int, n_models: int = 7, eps=0.1):
        learners = dict()
        for i in range(n_learners):
            learners[f"a{i}"] = river.ensemble.BaggingRegressor(
                model=river.preprocessing.StandardScaler() | river.linear_model.LinearRegression(optimizer=river.optim.SGD(lr=0.003)),
                n_models=n_models)
        self.eps = eps
        super().__init__(learners)

    def update(self, s, a, r):
        self.learners[a].learn_one({str(i): v for i, v in enumerate(s.squeeze())}, r)

    def choose(self, s: np.array):
        if np.random.random() > self.eps:
            values = {k: v.predict_one({str(i): v for i, v in enumerate(s.squeeze())}) for k, v in self.learners.items()}
            return max(values, key=values.get)
        return np.random.choice(list(self.learners.keys()))


class LinearExpertsLearner(BanditLearner):
    def __init__(self, n_learners: int, lrs: tuple = (0.005, 0.003, 0.001, 0.0007), eps=0.1):
        learners = dict()
        for i in range(n_learners):
            models = [
                river.preprocessing.StandardScaler() | river.linear_model.LinearRegression(optimizer=river.optim.SGD(lr=lr))
                for lr in lrs
            ]
            learners[f"a{i}"] = river.expert.EpsilonGreedyRegressor(models=models)
        self.eps = eps
        super().__init__(learners)

    def update(self, s, a, r):
        self.learners[a].learn_one({str(i): v for i, v in enumerate(s.squeeze())}, r)

    def choose(self, s: np.array):
        if np.random.random() > self.eps:
            values = {k: v.predict_one({str(i): v for i, v in enumerate(s.squeeze())}) for k, v in self.learners.items()}
            return max(values, key=values.get)
        return np.random.choice(list(self.learners.keys()))
