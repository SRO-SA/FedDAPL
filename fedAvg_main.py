import flwr as fl
import sys
import os
import ray
import uuid
import json
# ray.init(num_cpus=5)  # ← Choose enough for your number of clients
from utils import log_print
sys.path.append('/rhome/ssafa013/DGDDPM/DGDDPM/wilds')
log_print(sys.path)
# print("salam ", sy.TYPE_REGISTRY)  # shows all registered Syft objects
from DataLoader import OpenBHBDataset
from  server import Server, ServerDomainSpecHelper
from client import FedClient, FlowerClientWrapper
import time
from federated_data_splitter import FederatedOpenBHBSplitter
from strategy_custom import FedCustomStrategy
from flwr.server.strategy import FedAvg
import numpy as np
from client_factory import ClientFactory, ValidationClientFactory
bigdata_root = "/rhome/ssafa013/bigdata/raytmp"   # make sure this exists
os.makedirs(bigdata_root, exist_ok=True)
train_batch_size=3
num_domains = 5
NUM_ROUNDS = 15
# fedavg_wrapped.py  (replace your current fedavg_main.py)
from pathlib import Path
from flwr.common import parameters_to_ndarrays
import flwr as fl
import ray, sys, time, os
import torch
from typing import Dict, List, Tuple
from model import BrainCancer            # your model class
# --- project imports ---------------------------------------------------------
from utils import (log_print,                         # your logger
                   flatten_layer_param_list_for_flower,
                   flatten_layer_param_list_for_model,
                   reconstruct_layer_param_list)
from DataLoader import OpenBHBDataset
from server import Server, ServerDomainSpecHelper
from client_factory import ClientFactory, ValidationClientFactory
# -----------------------------------------------------------------------------

class FedAvgSave(FedAvg):
    """FedAvg that remembers its final aggregated parameters."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        ret = super().aggregate_fit(rnd, results, failures)
        if ret is not None:
            self.final_parameters = ret[0]        # (Parameters, metrics)
        return ret

def save_flower_weights(strategy: FedAvgSave,
                        ref_layers: list,
                        out_path: str) -> None:
    """Convert Flower Parameters → torch model → .pth."""
    nds      = parameters_to_ndarrays(strategy.final_parameters)
    weights  = reconstruct_layer_param_list(nds, ref_layers)
    parms = flatten_layer_param_list_for_model(weights)
    model = BrainCancer()
    # utility you already have – copies tensors
    model.receive_and_update_params(parms)
    torch.save(model.state_dict(), out_path)


# ------------------------------------------------------------------- helpers
def make_initial_parameters(server: Server) -> fl.common.Parameters:
    """Flatten the server’s dummy model so both FedAvg and your custom
    strategy start from *identical* weights."""
    from flwr.common import ndarrays_to_parameters
    flat = flatten_layer_param_list_for_flower(server.initial_dummy_paramters_list)
    return ndarrays_to_parameters(flat)

def make_on_fit_config_fn(batch_size: int, epochs: int):
    def fit_config(_round: int) -> Dict[str, int]:
        # sent to Client.fit()
        return {"batch_size": batch_size, "epochs": epochs}
    return fit_config

def weighted_mean(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Flower passes a list   [(num_samples_i, metrics_i), …].
       Compute weighted average for every metric key."""
    agg: Dict[str, float] = {}
    total = sum(num for num, _ in metrics)
    for num, m in metrics:
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + num * float(v)
    return {k: v / total for k, v in agg.items()}

def make_evaluate_fn(server: Server, batch_size: int = 1):
    """Run *exactly* the validation logic you already use inside
    FedCustomStrategy.evaluate()."""
    val_factory = ValidationClientFactory(server, server.val_dataset, model_type="Normal", batch_size=batch_size)
    ref_layers = server.initial_dummy_paramters_list          # structure template

    def evaluate(server_round: int,
                 parameters: List["np.ndarray"],
                 _config: Dict) -> Tuple[float, Dict[str, float]]:      # Flower-expected sig.

        # 1) rebuild layer structure from FedAvg’s flat ndarrays
        layer_params = reconstruct_layer_param_list(parameters, ref_layers)

        # 2) spin up an “offline” validation client with those weights
        val_client = val_factory(0, layer_params)
        logs = val_client.local_train_step(is_train=False)    # returns [(epoch, loss, auc)]

        # _, loss, auc = logs[-1]                               # take final epoch
        # (Optional) keep your old text log:
        with open(server.validation_log_path, "a") as f:
            if len(logs[0]) <=3:
                for epoch, loss, mae, site_mse, macro_mse in logs:
                    f.write(f"Round {server_round}: Loss={loss:.4f}, MAE={mae:.4f}, macro_mse={macro_mse:.4f}\n")
                    site_mse_str = '\n '.join([f'Site {k}: {v:.4f}' for k, v in site_mse.items()])
                    f.write(f"Site MSE:\n {site_mse_str}\n")  
                f.write(f"Validation Done\n\n")
            elif len(logs[0]) > 3:
                for epoch, loss, mae, site_mse, macro_mse in logs:
                    f.write(f"Round {server_round}: Loss={loss:.4f}, MAE={mae:.4f}, macro_mse={macro_mse:.4f}\n")
                    site_mse_str = '\n '.join([f'Site {k}: {v:.4f}' for k, v in site_mse.items()])
                    f.write(f"Site MSE:\n {site_mse_str}\n")                 
                f.write(f"Validation Done\n\n")
        return float(loss), {"loss": float(loss), "mae": float(mae), "macro_mse": float(macro_mse)}
    return evaluate
# ---------------------------------------------------------------------------


def run_federated_once(repeat_idx: int,
                       global_run_id: str) -> float:
    """A very thin wrapper around the original main-block code.
       Returns the last-round MAE."""
    # --------------- (BEGIN: copy/paste original body) -------------------
    _temp_dir = f"{bigdata_root}/{os.environ.get('SLURM_JOB_ID', str(time.time()))}/ray"
    namespace = f"flower_ns_{uuid.uuid4().hex[:8]}"
    ray.init(num_cpus=6, num_gpus=2, _temp_dir=_temp_dir,
             namespace=namespace, include_dashboard=False,
             ignore_reinit_error=True)

    splitter  = FederatedOpenBHBSplitter()
    clients_ds, val_ds, test_ds = splitter.get_federated_splits()

    run_id    = f"{global_run_id}/rep{repeat_idx+1}"
    server    = Server(num_clients=num_domains, val_dataset=val_ds,
                       test_dataset=test_ds, model_type="Normal",
                       run_id=run_id, num_rounds=NUM_ROUNDS)

    client_fn = ClientFactory(server.client_log_file_paths,
                              server, clients_ds,
                              model_type="Normal",
                              batch_size=train_batch_size, epochs=5)

    strategy = FedAvgSave(
        initial_parameters   = make_initial_parameters(server),
        on_fit_config_fn     = make_on_fit_config_fn(batch_size=3, epochs=5),
        evaluate_fn          = make_evaluate_fn(server, batch_size=1),
        fit_metrics_aggregation_fn       = weighted_mean,
        evaluate_metrics_aggregation_fn  = weighted_mean,
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=5, min_available_clients=5,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=5,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus":1,"num_gpus":0.5},
        ray_init_args={"_temp_dir":_temp_dir,"num_cpus":6,
                       "num_gpus":2,"include_dashboard":False,
                       "ignore_reinit_error":True,
                       "_system_config":{"worker_register_timeout_seconds":120}}
    )
    log_print("Training is done", context=f"MAIN-{repeat_idx+1}")
    # --------------- (END: original body) ---------------------------------

    # ---------- save checkpoint & return metric ---------------------------
    out_dir = Path("./runs")/f"run_{global_run_id}"
    # out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir/f"model_run{repeat_idx+1}.pth"
    save_flower_weights(strategy, server.initial_dummy_paramters_list, ckpt)

    last_rnd   = max(history.metrics_centralized.keys())
    # log_print(f"history: {history}\n")
    # log_print(f"history.metrics_center: {history.metrics_centralized.keys()}\n")

    mae     = history.metrics_centralized['mae'][-1][1]
    loss = history.metrics_centralized['loss'][-1][1]
    macro_mse = history.metrics_centralized['macro_mse'][-1][1]

    ray.shutdown()
    return loss, mae, macro_mse


if __name__ == "__main__":
    # _temp_dir = f"/scratch/ssafa013/{os.environ.get('SLURM_JOB_ID', str(time.time()))}/ray/"
    # _temp_dir = f"{bigdata_root}/{os.environ.get('SLURM_JOB_ID', str(time.time()))}/ray"

    # namespace = f"flower_ns_{uuid.uuid4().hex[:8]}"
    # ray.init(num_cpus=6,
    #         num_gpus=2,          # ⚠️ new line – make 2 GPUs visible
    #         _temp_dir= _temp_dir,
    #         namespace= namespace,
    #         include_dashboard=False,
    #         ignore_reinit_error=True )  # ← Choose enough for your number of clients
    # # ----- identical data split, logging paths, etc. ------------------------

    # ray_init_args = {
    #     "_temp_dir": _temp_dir,
    #     "num_cpus": 6,
    #     "num_gpus": 2,
    #     "include_dashboard": False,
    #     "ignore_reinit_error": True,
    #     "_system_config": {"worker_register_timeout_seconds": 120}
    # }
    # splitter  = FederatedOpenBHBSplitter()
    # clients_ds, val_ds, test_ds = splitter.get_federated_splits()
    # run_id    = time.strftime("%Y-%m-%d_%H-%M-%S")
    # server    = Server(num_clients=num_domains,
    #                    val_dataset=val_ds,
    #                    test_dataset=test_ds,
    #                    model_type="Normal",
    #                    run_id=run_id,
    #                    num_rounds=NUM_ROUNDS)

    # client_fn = ClientFactory(server.client_log_file_paths,
    #                           server,
    #                           clients_ds,
    #                           model_type="Normal",
    #                           batch_size=train_batch_size,
    #                           epochs=5)

    # # ------------- FedAvg, but with your hooks ------------------------------
    # strategy = fl.server.strategy.FedAvg(
    #     initial_parameters            = make_initial_parameters(server),
    #     on_fit_config_fn              = make_on_fit_config_fn(batch_size=3, epochs=5),
    #     evaluate_fn                   = make_evaluate_fn(server, batch_size=1),
    #     fit_metrics_aggregation_fn    = weighted_mean,
    #     evaluate_metrics_aggregation_fn = weighted_mean,
    #     fraction_fit                  = 1.0,   # keep identical sampling
    #     fraction_evaluate             = 0.0,   # we do server-side eval instead
    #     min_fit_clients               = 5,
    #     min_available_clients         = 5,
    # )

    # # ----------------------- run the simulation -----------------------------
    # fl.simulation.start_simulation(
    #     client_fn        = client_fn,
    #     num_clients      = 5,
    #     config           = fl.server.ServerConfig(num_rounds=15),
    #     strategy         = strategy,
    #     client_resources = {"num_cpus": 1, "num_gpus": 0.5},
    #     ray_init_args    = ray_init_args,
    # )

    # log_print("Training is done", context="MAIN")
    global_run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("./runs")/f"run_{global_run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    maes = []
    losses = []
    macro_mses = []
    for k in range(10):
        print(f"\n--- Federated experiment {k+1}/10 ---")
        loss, mae, macro_mse = run_federated_once(k, global_run_id)
        maes.append(mae)
        losses.append(loss)
        macro_mses.append(macro_mse)


    summary = out_dir/"summary.txt"
    with open(summary, "w") as f:
        f.write(f"Validation MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}\n")
        f.write(f"Validation Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}\n")
        f.write(f"Validation Macro MSE: {np.mean(macro_mses):.4f} ± {np.std(macro_mses):.4f}\n")

    print(f"\n✅  All federated runs finished. Summary → {summary}")
