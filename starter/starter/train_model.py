# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_slice_performance

# Add code to load in the data.
data = pd.read_csv('../data/cleaned_census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# Save encoders
with open('../model/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

with open('../model/label_binarizer.pkl', 'wb') as f:
    pickle.dump(lb, f)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
with open('../model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Model inference
y_test_pred = inference(model, X_test)

# Compute performance metrics
precision, recall, fbeta = compute_model_metrics(y_test, y_test_pred)
print('Performance metrics on test set:')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1:{fbeta:.4f}')

# Compute performance metrics for each slice
compute_slice_performance(model, test.reset_index(drop=True), X_test, y_test, cat_features)
