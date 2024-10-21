import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.fixture
def neg_example():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-states"
    }
    return data


@pytest.fixture
def pos_example():
    data = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-states"
    }
    return data


def test_get():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json()['output'] == 'Welcome!'


def test_post_with_pos_example(pos_example):
    r = client.post('/predict', json=pos_example)
    assert r.status_code == 200
    assert r.json()['output'] == '>50K'


def test_post_with_neg_example(neg_example):
    r = client.post('/predict', json=neg_example)
    assert r.status_code == 200
    assert r.json()['output'] == '<=50K'
