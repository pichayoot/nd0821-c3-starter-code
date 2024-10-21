from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pickle


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_performance(model, X_raw, X_encoded, y, cat_features, output_path='../slice_output.txt'):
    with open(output_path, 'w') as f:
        f.write(f'Categorical columns: {", ".join(cat_features)}\n')
        for slice_col in cat_features:
            f.write(f'\nComputing slice performances on {slice_col}:\n')
            for slice in X_raw[slice_col].unique():
                slice_idx = X_raw[X_raw[slice_col] == slice].index

                slice_preds = inference(model, X_encoded[slice_idx])
                precision, recall, f1 = compute_model_metrics(y[slice_idx], slice_preds)

                f.write(f'Slice: {slice}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1:{f1:.4f}\n')


def load_model_artifacts(model_path, encoder_path, lb_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    with open(lb_path, 'rb') as f:
        lb = pickle.load(f)

    return model, encoder, lb
