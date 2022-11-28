import os

import servicefoundry.core as sfy
from servicefoundry import Build, Job, PythonBuild, Resources, Manual



def deploy_job():
    sfy.login(api_key=os.getenv("TFY_API_KEY"))
    image = Build(
        build_spec=PythonBuild(
            command="python deploy_service.py",
            requirements_path="requirements.txt",
        )
    )
    job = Job(
        name="churn-train-publish-job",
        image=image,
        resources=Resources(
            memory_limit=2500, memory_request=2000, cpu_limit=4, cpu_request=3.5
        ),
        trigger=Manual(run=True),
        env={
            "TFY_HOST": os.getenv("TFY_HOST"),
            "TFY_API_KEY": os.getenv("TFY_API_KEY"),
            "WORKSPACE_FQN": os.getenv("WORKSPACE_FQN"),
        },
    )
    job.deploy(workspace_fqn=os.getenv("WORKSPACE_FQN"))
    


if __name__ == "__main__":
    deploy_job()
    print("Job deployed successfully")
