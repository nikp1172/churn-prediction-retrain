from batch_infer import monitor
from model import prepare_model
from deploy_service import deploy_model


if __name__ == "__main__":
    fqn = prepare_model()
    deploy_model(fqn)
    monitor(fqn)


