import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from starter.ml.model import train_model, inference, compute_model_metrics


@pytest.fixture
def X():
    np.random.seed(0)
    return np.random.rand(10, 5)


@pytest.fixture
def y():
    np.random.seed(0)
    return np.random.randint(0, 2, size=10)


@pytest.fixture
def model(X, y):
    return train_model(X, y)


@pytest.fixture
def y_pred(model, X):
    return inference(model, X)


def test_train_model(model):
    assert isinstance(model, RandomForestClassifier)


def test_inference(y_pred, X):
    assert len(y_pred) == X.shape[0]
    assert (y_pred >= 0).all() and (y_pred <= 1).all()


def test_compute_metrics(y, y_pred):
    precision, recall, f1 = compute_model_metrics(y, y_pred)
    assert 0 <= precision <= 1 and 0 <= recall <= 1 and 0 <= f1 <= 1
