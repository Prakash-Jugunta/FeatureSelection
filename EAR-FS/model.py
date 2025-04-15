import torch
import torch.nn as nn
import numpy as np
import logging
import pyswarms as ps
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FeatureSelectionMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.selection_rate = nn.Parameter(torch.ones([1, n_features]))

    def get_selection_rate(self):
        return torch.sigmoid(self.selection_rate)

    def forward(self, x):
        return x * torch.sigmoid(self.selection_rate)

class Classifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_features, int(n_features * 4)),
            nn.ReLU(),
            nn.Linear(int(n_features * 4), int(n_features * 2)),
            nn.ReLU(),
            nn.Linear(int(n_features * 2), n_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class AttentionModel(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.classifier = Classifier(n_features, n_classes)
        self.feature_selection_MLP = FeatureSelectionMLP(n_features)

    def forward(self, x):
        return self.classifier(self.feature_selection_MLP(x))

def select_feature(origin_f, select_rate, threshold):
    (a, l) = select_rate.shape
    flag = False
    x = None
    for i in range(l):
        if float(select_rate[0, i]) > threshold:
            if not flag:
                x = origin_f[:, i]
                flag = True
            else:
                x = torch.vstack([x, origin_f[:, i]])
    return x.permute(1, 0) if x is not None else origin_f[:, :1].permute(1, 0)

def sorted_idx(select_rate):
    a = select_rate[0, :]
    idx = sorted(range(len(a)), key=lambda k: a[k], reverse=True)
    return idx

def select_idx_F(origin_f, idx, num):
    x = origin_f[:, idx[0]]
    for i in range(1, min(num, len(idx))):
        x = torch.vstack([x, origin_f[:, idx[i]]])
    return x.permute(1, 0)

def add_one_features(origin, inp, idx, n):
    x = inp.permute(1, 0)
    x = torch.vstack([x, origin[:, idx[n-1]]])
    return x.permute(1, 0)

def add_more_features(origin, inp, idx, n_max, n):
    x = inp.permute(1, 0)
    for i in range(n, n_max):
        x = torch.vstack([x, origin[:, idx[i]]])
    return x.permute(1, 0)

def log(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = path
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def pso_fitness(particles, X, y, clf, validation_split=0.3):
    n_particles = particles.shape[0]
    fitness = np.zeros(n_particles)
    for i in range(n_particles):
        selected_features = particles[i] > 0.5
        if np.sum(selected_features) == 0:
            fitness[i] = 0
            continue
        X_subset = X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=validation_split, random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        fitness[i] = accuracy_score(y_test, y_pred)
    return -fitness  # Negative for minimization