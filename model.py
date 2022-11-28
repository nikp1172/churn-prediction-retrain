from luciferml.supervised.classification import Classification
import pandas as pd
from deploy_job import experiment_track

def train_model():
    df = pd.read_csv("Data/Churn_Modelling.csv")
    X = df.iloc[:, 3:-1].drop(["Geography", "Gender"], axis=1)
    y = df.iloc[:, -1]
    classifier = Classification(predictor=["rfc"])
    classifier.fit(X, y)
    return classifier.classifier, X, y

def prepare_model():
    model, features, labels = train_model()
    experiment_track(model, features, labels)