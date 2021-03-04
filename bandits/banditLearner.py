import numpy as np
from sklearn.linear_model import SGDRegressor
import xgboost as xgb


class BanditLearner():
    """Class that has models for every arm and makes decsions based on the best outcome

    Offers the following API:
    - choose(s: np.array) -> str   # Chooses the arm to pick for that state (context)
    - update(s, a, r) -> None      # Updates the arm with state and reward
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
    def __init__(self, n_learners: int, n_trees: int):
        learners = dict()
        for i in range(n_learners):
            learners[f"a{i}"] = xgb.XGBRegressor(n_estimators=n_trees)
        super().__init__(learners)

    def update(self, s, a, r):
        self.learners[a].fit(s, np.array(r))