from luciferml.supervised.classification import Classification
import pandas as pd
import mlfoundry as mlf


def experiment_track(model, features, labels):
    mlf_api = mlf.get_client()
    mlf_run = mlf_api.create_run(
        project_name="churn-train-job", run_name="churn-train-job-1"
    )
    fqn = mlf_run.log_model(
        name="Best_Model",
        model=model,
        framework=mlf.ModelFramework.SKLEARN,
        description="My_Model",
    )
    mlf_run.log_dataset("features", features)
    mlf_run.log_dataset("labels", labels)
    return fqn.fqn


def train_model():
    df = pd.read_csv("Data/Churn_Modelling.csv")
    X = df.iloc[:, 3:-1].drop(["Geography", "Gender"], axis=1)
    y = df.iloc[:, -1]
    classifier = Classification(predictor=["rfc"])
    classifier.fit(X, y)
    return classifier.classifier, X, y


def prepare_model():
    model, features, labels = train_model()
    fqn = experiment_track(model, features, labels)
    return fqn