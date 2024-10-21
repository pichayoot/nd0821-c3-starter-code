# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a binary-classification random forest model that predicts whether an individual has income higher than $50,000. 
The features that are required for the model are age, workclass, fnlgt, education (both string and numeric), marital status,
occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country.

## Intended Use
The intended use case for this model is to complete a project on Udacity Machine Learning DevOps Nanodegree program.

## Training Data
The model was trained on the UCI census income dataset (found here: https://archive.ics.uci.edu/dataset/20/census+income).
The data was randomly split so that 80% of the dataset was used as a training dataset.

## Evaluation Data
The data was randomly split so that 20% of the dataset was used as a held-out test set.

## Metrics
The performance of the model on a held-out test set is as follows:
- Precision: 0.7767
- Recall: 0.5356
- F1-score: 0.6340

## Ethical Considerations
Please do not use the result of this model to make decision about individual salary.

## Caveats and Recommendations
There are imbalances in the dataset, especially the race feature. 