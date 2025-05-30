"""embeddedexample: A Flower / PyTorch app."""

import torch, io, os
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import subprocess

from embeddedexample.task import (
    FeatureLearningModule,
    get_weights,
    load_data_from_disk,
    set_weights,
    test,
    train,
    FEDMALDETECT,
    Classifier
)

class FlowerClient(NumPyClient):
    def __init__(self, context, trainloader, local_epochs, learning_rate, input_dim, hidden_dim, proj_dim, output_dim):
        self.featurelearningmodule = FeatureLearningModule(input_dim, hidden_dim, proj_dim)
        self.classifier = Classifier(hidden_dim, hidden_dim, output_dim)
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.context = context

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.featurelearningmodule, parameters)
        train(
            self.featurelearningmodule,
            self.trainloader,
            self.local_epochs,
            self.lr
        )
        return get_weights(self.featurelearningmodule), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.featurelearningmodule, parameters)
        ft_bytes = config["finetuned_model"]
        buf = io.BytesIO(ft_bytes)
        ft_state_dict = torch.load(buf)
        self.classifier.load_state_dict(ft_state_dict)
        del ft_state_dict
        final_model = FEDMALDETECT(input_dim=115, hidden_dim=128, output_dim=2, global_feature_learning_model=self.featurelearningmodule, classifier=self.classifier)
        X_test, y_test = load_data_from_disk(self.context.node_config["labeled_data"], self.context.run_config["server-batch-size"], True, False)
        precision, recall,f1, roc_auc, pr_auc, cm = test(final_model, X_test, y_test)
        return 0.0, len(y_test), {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm
        }

def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    dataset_path = context.node_config["dataset-path"]
    batch_size = context.run_config["batch-size"]
    trainloader = load_data_from_disk(dataset_path, batch_size, False, False)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    input_dim=115
    hidden_dim=128
    proj_dim=64
    output_dim = 2
    return FlowerClient(context, trainloader, local_epochs, learning_rate, input_dim, hidden_dim, proj_dim, output_dim).to_client()

app = ClientApp(client_fn)

@app.lifespan()
def lifespan(context: Context) -> None:
    pid = os.getpid()
    print("Process ID: ",pid)
    log_file = "client_usage.log"
    pidstat_cmd = [
        "pidstat",
        "-p", f"{pid}",
        "-r",
        "-u",
        "-h",
        "1",
    ]
    print("Starting resource monitoring...")
    monitor_proc = subprocess.Popen(pidstat_cmd, stdout=open(log_file, "a"), stderr=subprocess.STDOUT)

    yield

    print("Stopping resource monitoring...")
    monitor_proc.terminate()
    try:
        monitor_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        monitor_proc.kill()

