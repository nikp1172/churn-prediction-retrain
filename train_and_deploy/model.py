import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as Classification
import mlfoundry as mlf
from sklearn.metrics import accuracy_score


def experiment_track(model, params, metrics, features):
    mlf_api = mlf.get_client()
    mlf_run = mlf_api.create_run(
        project_name="churn-train", run_name="churn-train-job-1"
    )
    mlf_run.log_params(params)
    mlf_run.log_metrics(metrics)
    mlf_run.log_dataset("features", features)
    model_version = mlf_run.log_model(
        name="churn-model",
        model=model,
        framework=mlf.ModelFramework.SKLEARN,
        description="My_Model",
        model_schema={
            "features": [
                {"name": "CreditScore", "type": "float"},
                {"name": "Age", "type": "float"},
                {"name": "Tenure", "type": "float"},
                {"name": "Balance", "type": "float"},
                {"name": "NumOfProducts", "type": "float"},
                {"name": "HasCrCard", "type": "float"},
                {"name": "IsActiveMember", "type": "float"},
                {"name": "EstimatedSalary", "type": "float"}
            ],
            "prediction": "categorical"
        },
        custom_metrics=[{"name": "log_loss", "type": "metric", "value_type": "float"}]
    )
    return model_version.fqn


def train_model():
    df = pd.read_csv("Data/Churn_Modelling.csv")
    X = df.iloc[:, 3:-1].drop(["Geography", "Gender"], axis=1)
    y = df.iloc[:, -1]
    classifier = Classification()
    classifier.fit(X, y)
    metrics = {
        "accuracy": accuracy_score(y, classifier.predict(X))
    }
    return classifier, classifier.get_params(), metrics, X


def prepare_model():
    model, params, metrics, X = train_model()
    fqn = experiment_track(model, params, metrics, X)
    return fqn