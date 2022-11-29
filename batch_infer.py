import os
import pandas as pd
import uuid
import time
from datetime import datetime

import mlfoundry

client = mlfoundry.get_client()


def load_model(fqn):
    return client.get_model(fqn).load()


def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/d4rk-lucif3r/Automated-Model-Training-and-Deployment-Pipeline-with-Github-Actions/main/Data/Churn_Modelling.csv")
    df = df.sample(n=20)
    y = df.iloc[:, -1]
    X = df.iloc[:, 3:-1].drop(["Geography", "Gender"], axis=1)
    return X, y


def monitor(fqn):
    model = load_model(fqn)
    X, y_actual = load_data()
    y_pred = model.predict(X)

    print(f"Len: X {len(X)}, y_actual {len(y_actual)}, y_pred {len(y_pred)}")

    data_ids = []

    print("Logging predictions ...")
    for (_, row), prediction in zip(X.iterrows(), y_pred):
        data_id = uuid.uuid4().hex
        prediction_value = str(prediction)
        data_ids.append(data_id)
        client.log_predictions(
            model_version_fqn=fqn,
            predictions=[
                mlfoundry.Prediction(
                    data_id=data_id,
                    features=row.to_dict(),
                    prediction_data={
                        "value": prediction_value,
                    },
                    occurred_at=datetime.utcnow(),
                )
            ],
        )

    time.sleep(10)

    print("Logging actuals ...")
    for data_id, actual in zip(data_ids, y_actual):
        actual_value = str(actual)
        client.log_actuals(
            model_version_fqn=fqn,
            actuals=[mlfoundry.Actual(data_id=data_id, value=actual_value)],
        )
    print("All Done!")
