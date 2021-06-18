from collections import OrderedDict
import logging
import os
import sys
sys.path.append(os.path.join(os.getcwd(), os.pardir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import SimpleExpSmoothing

from bandits.arms import GaussianMixtureArm
from bandits.context import Context
from bandits.banditPlayer import BanditPlayer
from bandits.banditLearner import (SGDLearner, XGBLearner, OptimisticSGDLearner, AdaptiveRandomForestLearner,
                                   PerceptronLearner, BaggedLinearRegressor ,LinearExpertsLearner)


diag_down = np.array([[-1, 1], [1, -1]])
diag_up = np.array([[-1, -1], [1, 1]])
left = np.array([[-1, -1], [-1, 1]])
right = np.array([[1, -1], [1, 1]])
top = np.array([[-1, 1], [1, 1]])
bottom = np.array([[-1, -1], [1, -1]])


n = 10000
context = Context(n, 2)
contexts = context.contexts
a0 = GaussianMixtureArm(
    centres = np.array(diag_up),
    stds= np.array([1, 1]),
    factor=1,
    noise=.05,
)


def make_audit_data():
    N = 100
    margins = np.linspace(-2, 2, N).reshape([-1, 1])
    px = np.tile(margins, [N, 1])
    py = np.repeat(np.flip(margins), N).reshape([-1, 1])
    X = np.concatenate([px, py], axis=1)
    return X

def reshape_vals(vals):
    n = int(round(len(vals)**0.5, 0))
    return np.array(vals).reshape([n, n])

def plot_arm(arm):
    values = arm.value(make_audit_data())[1]
    plt.imshow(reshape_vals(values))

def polyise(v):
    return poly.fit_transform(np.array(v).reshape([1, -1]))

def dictise(v):
    return {i: v for i, v in enumerate(polyise(v)[0])}

poly = PolynomialFeatures(4)
Xc = poly.fit_transform(contexts)
Xa = poly.fit_transform(make_audit_data())
a0.ev(np.array([[-0, 0]]))




from river import compose
from river import linear_model
from river import metrics
from river import evaluate
from river import preprocessing
from river import optim
from river import stream
import river



model = preprocessing.StandardScaler()
model |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))

s2 = stream.iter_array(Xa)
audit = [model.predict_one(i[0]) for i in s2]
plt.imshow(reshape_vals(audit))

s1 = stream.iter_array(Xc, a0.ev(contexts))
for c, v in s1:
    model.learn_one(c, v)

s2 = stream.iter_array(Xa)
audit = [model.predict_one(i[0]) for i in s2]
plt.imshow(reshape_vals(audit))

metric = metrics.RMSE()
evaluate.progressive_val_score(stream.iter_array(Xc, a0.ev(contexts)), model, metric, print_every=int(Xc.shape[0]/20))

s2 = stream.iter_array(Xa)
audit = [model.predict_one(i[0]) for i in s2]
plt.imshow(reshape_vals(audit))

plt.imshow(reshape_vals(audit))
plot_arm(a0)

# can linearexpertslearners learn?
lel = LinearExpertsLearner(2).learners["a0"]

s2 = stream.iter_array(Xa)
audit = [lel.predict_one(i[0]) for i in s2]
plt.imshow(reshape_vals(audit))

s1 = stream.iter_array(Xc, a0.ev(contexts))
for c, v in s1:
    lel.learn_one(c, v)

s2 = stream.iter_array(Xa)
audit = [lel.predict_one(i[0]) for i in s2]
plt.imshow(reshape_vals(audit))
# --> no

# can one component of them learn?
lel2 = river.preprocessing.StandardScaler() | river.linear_model.LinearRegression(optimizer=river.optim.SGD(lr=0.003))

s2 = stream.iter_array(Xa)
audit = [lel2.predict_one(i[0]) for i in s2]
plt.imshow(reshape_vals(audit))

s1 = stream.iter_array(Xc, a0.ev(contexts))
for c, v in s1:
    lel2.learn_one(c, v)

s2 = stream.iter_array(Xa)
audit = [lel2.predict_one(i[0]) for i in s2]
plt.imshow(reshape_vals(audit))
# also no