"""embeddedexample: A Flower / PyTorch app."""

import torch
import io, json, os
import subprocess, time
from typing import List, Tuple, Optional
from flwr.common import Context, Metrics, ndarrays_to_parameters, Parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, EvaluateIns, EvaluateRes
from embeddedexample.task import FeatureLearningModule, get_weights, feddyn, FEDMALDETECT, Classifier, server_finetuning, load_data_from_disk, test

class FedDynStrategy(FedAvg):
    def __init__(self, *, global_feature_learning_model, beta: float, total_rounds: int, context: Context, **kwargs) -> None:
        super().__init__(**kwargs)
        self.global_feature_learning_model = global_feature_learning_model
        self.classifier = None
        self.beta = beta
        self.total_rounds = total_rounds
        self.context = context
        self._last_finetuned_model_bytes = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy | FitRes]],
        failures: List[Tuple[ClientProxy | FitRes] | BaseException],
    ):
        start_time = time.time()
        if not results:
            return None
        client_models = []
        client_data_sizes = []
        for _, fit_res in results:
            client_data_sizes.append(fit_res.num_examples)
            nds = parameters_to_ndarrays(fit_res.parameters)
            m = FeatureLearningModule(
                self.global_feature_learning_model.input_dim,
                self.global_feature_learning_model.hidden_dim,
                self.global_feature_learning_model.proj_dim,
            )
            keys = list(m.state_dict().keys())
            sd = {k: torch.tensor(v) for k, v in zip(keys, nds)}
            m.load_state_dict(sd)
            client_models.append(m)

        self.global_feature_learning_model = feddyn(client_models, client_data_sizes, self.global_feature_learning_model, self.beta)
        new_nds = get_weights(self.global_feature_learning_model)
        new_params = ndarrays_to_parameters(new_nds)
        train_loader, X_test, y_test = load_data_from_disk('/home/ubuntu/Client-5/data.csv', self.context.run_config["server-batch-size"], True, True)
        final_model = FEDMALDETECT(input_dim=115, hidden_dim=128, output_dim=2, global_feature_learning_model=self.global_feature_learning_model, classifier=self.classifier)
        self.classifier = server_finetuning(final_model, train_loader, X_test, y_test)
        buf = io.BytesIO()
        torch.save(self.classifier.state_dict(), buf)
        ft_bytes = buf.getvalue()
        self._last_finetuned_model_bytes = ft_bytes
        # torch.save(
        #     self.global_feature_learning_model.state_dict(),
        #     "global_feature_learning_model_final_round.pth")
        # print(f"Saved final global model → global_feature_learning_model_final_round.pth")
        end_time = time.time()
        server_side_time = end_time-start_time
        print(f"Server side time: {server_side_time:.2f} seconds")

        return new_params, {"finetuned_model": ft_bytes}

    def configure_evaluate(
        self,
        server_round: int,
        parameters,
        client_manager,) -> list[tuple[ClientProxy, EvaluateIns]]:
    
        clients = client_manager.sample(
            num_clients=len(client_manager.all()),
            min_num_clients=len(client_manager.all()),
        )

        # pull out the finetuned bytes we stored
        model_bytes = self._last_finetuned_model_bytes
        if model_bytes is None:
            raise RuntimeError(
                "FedDynStrategy: _last_finetuned_model_bytes is missing; "
                "did aggregate_fit run?"
            )

        # Build a config dict that includes your finetuned‐model bytes
        cfg = {"finetuned_model": model_bytes}

        # Create an EvaluateIns that carries both the global weights AND your bytes
        evaluate_ins = EvaluateIns(parameters, cfg)

        # Return (client, EvaluateIns) pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[BaseException],) -> tuple[float, dict[str, float]]:
        if not results:
            return 0.0, {}
        # Initialize accumulators
        total_examples = 0
        aggregated_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
        }

        for _, eval_res in results:
            num_examples = eval_res.num_examples
            total_examples += num_examples
            for key in aggregated_metrics:
                aggregated_metrics[key] += eval_res.metrics[key] * num_examples

        # Weighted average
        for key in aggregated_metrics:
            aggregated_metrics[key] /= total_examples

        aggregated_metrics["round"] = server_round

        print(aggregated_metrics)
        with open("results.json", "w") as json_file:
            json.dump(aggregated_metrics, json_file, indent=4)


        # Return dummy loss and the aggregated metrics
        return 0.0, aggregated_metrics

def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_rounds = context.run_config["num-server-rounds"]
    input_dim=115
    hidden_dim=128
    proj_dim=64
    beta = 0.1
    global_feature_learning_model = FeatureLearningModule(input_dim, hidden_dim, proj_dim)
    ndarrays = get_weights(global_feature_learning_model)
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedDynStrategy(
        global_feature_learning_model=global_feature_learning_model,
        beta=beta,
        total_rounds=num_rounds,
        context=context,
        fraction_fit=0.5,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=4, #4
        min_fit_clients=2, #2
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds,round_timeout=180000)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)

@app.lifespan()
def lifespan(context: Context) -> None:
    pid = os.getpid()
    print("Process ID: ",pid)
    log_file = "server_usage.log"
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
