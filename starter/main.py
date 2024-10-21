# Put the code for your API here.
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from starter.ml.model import load_model_artifacts, inference
from starter.ml.data import process_data
import pandas as pd
import os

app = FastAPI()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(ROOT_DIR, 'model', 'model.pkl')
encoder_path = os.path.join(ROOT_DIR, 'model', 'encoder.pkl')
lb_path = os.path.join(ROOT_DIR, 'model', 'label_binarizer.pkl')
model, encoder, lb = load_model_artifacts(model_path, encoder_path, lb_path)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class Item(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


@app.get("/")
def root():
    return {"output": "Welcome!"}


@app.post("/predict")
def read_item(item: Item):
    raw_features = pd.DataFrame(jsonable_encoder(item), index=[0])
    raw_features.columns = [x.replace('_', '-') for x in raw_features.columns]
    encoded_features, _, _, _ = process_data(raw_features, cat_features, training=False, encoder=encoder, lb=lb)
    pred = lb.inverse_transform(inference(model, encoded_features))[0]
    output = {'output': str(pred)}
    return output
