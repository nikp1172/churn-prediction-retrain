import os
from servicefoundry import Build, Job, PythonBuild, Resources, Manual


def deploy_job():
    image = Build(
        build_spec=PythonBuild(
            command="python train_and_deploy.py",
            requirements_path="requirements.txt",
        )
    )
    job = Job(
        name="churn-train-deploy",
        image=image,
        resources=Resources(
            memory_limit=2500, memory_request=2000, cpu_limit=1, cpu_request=1
        ),
        trigger=Manual(run=True),
        env={
            "TFY_HOST": "https://app.truefoundry.com",
            "TFY_API_KEY": os.getenv("TFY_API_KEY"),
            "WORKSPACE_FQN": os.getenv("WORKSPACE_FQN"),
        },
    )
    job.deploy(workspace_fqn=os.getenv("WORKSPACE_FQN"))
    

if __name__ == "__main__":
    deploy_job()
    print("Job deployed successfully")
