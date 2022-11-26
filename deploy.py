import os

import mlfoundry as mlf
import servicefoundry.core as sfy
from servicefoundry import Build, Job, PythonBuild, Resources, Schedule


def experiment_track(model, features, labels):
    mlf_api = mlf.get_client()
    mlf_run = mlf_api.create_run(
        project_name='churn-train-job', run_name='churn-train-job-1')
    fn = mlf_run.log_model(name='Best_Model', model=model,
                           framework=mlf.ModelFramework.SKLEARN, description='My_Model')
    mlf_run.log_dataset("features", features)
    mlf_run.log_dataset("labels", labels)


def deploy_job():
    sfy.login(api_key = os.getenv('TFY_API_KEY'))
    image = Build(
        build_spec=PythonBuild(
            command="python service.py",
            requirements_path="requirements.txt",
        )
    )
    job = Job(
        name="churn-train-publish-job",
        image=image,
        resources=Resources(memory_limit=2500, memory_request=2000,
                            cpu_limit=4, cpu_request=3.5),
        trigger=Schedule(
            schedule="0 9 * * *",
            concurrency_policy="Forbid"
        ),
        env={"TFY_HOST": os.getenv('TFY_HOST'), "TFY_API_KEY": os.getenv(
            'TFY_API_KEY'), "WORKSPACE_FQN": os.getenv('WORKSPACE_FQN')}
    )
    job.deploy(workspace_fqn=os.getenv('WORKSPACE_FQN'))


if __name__ == "__main__":
    deploy_job()
    print('Job deployed successfully')
