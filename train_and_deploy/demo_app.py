import numpy as np
import pandas as pd
import gradio as gr
import mlfoundry as mlf
import servicefoundry.core as sfy
import datetime
import os

sfy.login(api_key=os.getenv("TFY_API_KEY"))

mlf_client = mlf.get_client()
runs = mlf_client.get_all_runs("churn-train")
run = mlf_client.get_run(runs["run_id"][0])

model = mlf_client.get_model(
    os.getenv('MODEL_VERSION_FQN')
)
model_schema = model.model_schema
model = model.load()


df = run.get_dataset("features")
df = pd.DataFrame(df.features)

inputs = []
i = 0
sample = df.iloc[0:1].values.tolist()[0]
for x in df.columns:
    if df[x].dtype == "object":
        inputs.append(gr.Textbox(label=x, value=sample[i]))
    elif df[x].dtype == "float64" or df[x].dtype == "int64":
        inputs.append(
            gr.Number(label=x, value=sample[i]),
        )
    i += 1


def predict(*val):
    print(val)
    global model
    if type(val) != list:
        val = [val]
    if type(val) != np.array:
        print("conv")
        val = np.array(val)
        print(val.shape)
    if val.ndim == 1:
        print("reshape")
        val = val.reshape(1, -1)
    pred = model.predict(val)
    return pred.tolist()[0]


desc = f"""## Model Deployed at {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}"""

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=[gr.Textbox(label="Churn")],
    description=desc,
    title="Churn Predictor",
)
app.launch(server_name="0.0.0.0", server_port=8080)