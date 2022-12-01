import os
import servicefoundry.core as sfy
from servicefoundry import Build, PythonBuild, Resources, Service


def deploy_model(fqn):
    sfy.login(api_key=os.getenv("TFY_API_KEY"))
    service = Service(
        name="churn-trained-from-job",
        image=Build(
            build_spec=PythonBuild(
                command="python demo_app.py",
            ),
        ),
        ports=[{"port": 8080}],
        resources=Resources(
            memory_limit=1500, memory_request=1000, cpu_limit=1, cpu_request=0.5
        ),
        env={
            "TFY_API_KEY": os.getenv("TFY_API_KEY"),
            "WORKSPACE_FQN": os.getenv("WORKSPACE_FQN"),
            "MODEL_VERSION_FQN": fqn
        },
    )
    service.deploy(workspace_fqn=os.getenv("WORKSPACE_FQN"))



