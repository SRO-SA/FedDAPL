import flwr as fl
import sys
import os
import ray
import uuid
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
from utils import debug_function, log_print


import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Parameters

class TrustFedScaffold(fl.server.strategy.FedAvg):
    def __init__(
        self,
        ref_layers,
        max_norm: float = 5.0,
        initial_parameters: fl.common.Parameters = None,
        **kwargs,
    ):
        """
        ref_layers: reference list of layer tensors (same you pass to
                    reconstruct_layer_param_list).  Used to know how many
                    ndarrays belong to θ versus c.
        """
        # pass initial_parameters on to FedAvg
        super().__init__(initial_parameters=initial_parameters, **kwargs)
        self.n_ref = len(ref_layers)  # number of ndarrays in θ
        self.c_global = [np.zeros_like(p) for p in ref_layers]
        self.max_norm = max_norm
        # --- seed the trust‑region centre right away ---
        self._current_params = initial_parameters
        
    def initialize_parameters(self, client_manager):
        self._current_params = super().initialize_parameters(client_manager)
        return self._current_params

    # ---------------- broadcast -----------------
    def parameters(self) -> Parameters:
        """Ship θ  +  c  as one long list."""
        nd = parameters_to_ndarrays(super().parameters())   # θ
        combo = nd + self.c_global                         # concat
        return ndarrays_to_parameters(combo)


    def aggregate_fit(self, rnd, results, failures):
        # Bail out if no one answered
        if len(results) == 0:
            return self.parameters(), {}

        # split weights / ci on every client -------------------------
        theta_list, ci_new_list = [], []
        for _, fit_res in results:
            combo = parameters_to_ndarrays(fit_res.parameters)
            theta_list.append(combo[: self.n_ref])
            ci_new_list.append(combo[self.n_ref :])
            
        # ---------- normal FedAvg (optionally clip) on θ ------------
        current = parameters_to_ndarrays(self.parameters())[: self.n_ref]
        clipped = []
        for nd_theta in theta_list:
            delta = [w - g for w, g in zip(nd_theta, current)]
            norm = np.sqrt(sum((d**2).sum() for d in delta))
            if norm > self.max_norm:
                delta = [d * (self.max_norm / (norm + 1e-12)) for d in delta]
            clipped.append([g + d for g, d in zip(current, delta)])

        # weighted mean (we ignore sample counts == uniform clients)
        new_theta = [
            sum(layer_k) / len(clipped) for layer_k in zip(*clipped)
        ]

        # ---------- SCAFFOLD global control update ------------------
        # ci_new_list and self.c_global are both length-n_ref lists
        avg_ci_new = [
            sum(layer_k) / len(ci_new_list) for layer_k in zip(*ci_new_list)
        ]
        self.c_global = [
            c + (c_new - c) for c, c_new in zip(self.c_global, avg_ci_new)
        ]

        # Store back as Parameters so Flower keeps rolling
        combo_new = new_theta + self.c_global
        self._parameters = ndarrays_to_parameters(combo_new)
        return self._parameters, {}





# =================================================================================


train_batch_size=3
num_domains = 5
NUM_ROUNDS = 15
# fedavg_wrapped.py  (replace your current fedavg_main.py)

import ray, sys, time, os
import torch
from typing import Dict, List, Tuple, Any

# --- project imports ---------------------------------------------------------
from utils import (log_print,                         # your logger
                   flatten_layer_param_list_for_flower,
                   reconstruct_layer_param_list)
from DataLoader import OpenBHBDataset
from server import Server, ServerDomainSpecHelper
from client_factory import ClientFactory, ValidationClientFactory
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------- helpers
@debug_function(context="MAIN")
def make_initial_parameters(server: Server) -> fl.common.Parameters:
    """Flatten the server’s dummy model so both FedAvg and your custom
    strategy start from *identical* weights."""
    from flwr.common import ndarrays_to_parameters
    flat = flatten_layer_param_list_for_flower(server.initial_dummy_paramters_list)
    return ndarrays_to_parameters(flat)

@debug_function(context="MAKE FIT CONF")
def make_on_fit_config_fn(batch_size: int, epochs: int):
    def fit_config(_round: int) -> Dict[str, int]:
        # sent to Client.fit()
        return {"batch_size": batch_size, "epochs": epochs, "current_round": _round, "num_rounds": NUM_ROUNDS}
    return fit_config

@debug_function(context="AVERAGE")
def weighted_mean(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Flower passes a list   [(num_samples_i, metrics_i), …].
       Compute weighted average for every metric key."""
    agg: Dict[str, float] = {}
    total = sum(num for num, _ in metrics)
    for num, m in metrics:
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + num * float(v)
    return {k: v / total for k, v in agg.items()}

@debug_function(context="MAKE EVALUATE CONF")
def make_on_evaluate_config_fn(val_batch_size: int):
    def evaluate_config(server_round: int) -> Dict[str, Any]:
        return {"batch_size": val_batch_size, "epochs": 1, "current_round": server_round, "num_rounds": NUM_ROUNDS}

    return evaluate_config
# -----------------------------------------------------------------------------

@debug_function(context="MAKE EVALUATE FN")
def make_evaluate_fn(server: Server, batch_size: int = 1, num_rounds: int = NUM_ROUNDS):
    """Run *exactly* the validation logic you already use inside
    FedCustomStrategy.evaluate()."""
    val_factory = ValidationClientFactory(server, server.val_dataset, model_type="DANN3D", batch_size=batch_size)
    ref_layers = server.initial_dummy_paramters_list          # structure template

    def evaluate(server_round: int,
                 parameters: List["np.ndarray"],
                 _config: Dict) -> Tuple[float, Dict[str, float]]:      # Flower-expected sig.
        # if server_round == 0:
        #     return 0.0, {"loss": 0.0, "acc": 0.0}  # no validation on round 0
        # 1) rebuild layer structure from FedAvg’s flat ndarrays
        layer_params = reconstruct_layer_param_list(parameters, ref_layers)
        _config["current_round"] = server_round
        _config["num_rounds"] = num_rounds
        log_print(f" The config is {_config}", context="EVAL")
        # 2) spin up an “offline” validation client with those weights
        val_client = val_factory(0, layer_params)
        logs = val_client.local_train_step(is_train=False, config=_config)    # returns [(epoch, loss, acc)]

        # _, loss, acc = logs[-1]                               # take final epoch
        # (Optional) keep your old text log:
        with open(server.validation_log_path, "a") as f:
            if len(logs[0]) <=3:
                for epoch, loss, acc in logs:
                    f.write(f"Round {server_round}: Loss={loss:.4f}, acc={acc:.4f}\n")
                f.write(f"Validation Done\n\n")
            elif len(logs[0]) > 3:
                for epoch, loss, acc, d_loss, p_loss in logs:
                    f.write(f"Round {server_round}: Loss={loss:.4f}, acc={acc:.4f}, Domain Loss={d_loss:.4f}, Prediction Loss={p_loss:.4f}\n")
                f.write(f"Validation Done\n\n")
        return float(loss), {"loss": float(loss), "acc": float(acc)}
    return evaluate
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    _temp_dir = f"/scratch/ssafa013/{os.environ.get('SLURM_JOB_ID', str(time.time()))}/ray/"
    namespace = f"flower_ns_{uuid.uuid4().hex[:8]}"
    ray.init(num_cpus=5,
            _temp_dir= _temp_dir,
            namespace= namespace,
            include_dashboard=False,
            ignore_reinit_error=True )  # ← Choose enough for your number of clients
    # ----- identical data split, logging paths, etc. ------------------------
    splitter  = FederatedOpenBHBSplitter()
    clients_ds, val_ds, test_ds = splitter.get_federated_splits()
    run_id    = time.strftime("%Y-%m-%d_%H-%M-%S")
    server    = Server(num_clients=num_domains,
                       val_dataset=val_ds,
                       test_dataset=test_ds,
                       model_type="DANN3D",
                       run_id=run_id,
                       num_rounds=NUM_ROUNDS)

    client_fn = ClientFactory(server.client_log_file_paths,
                              server,
                              clients_ds,
                              model_type="DANN3D",
                              batch_size=train_batch_size,
                              epochs=5)

    # ------------- FedAvg, but with your hooks ------------------------------
    strategy = TrustFedScaffold(
        ref_layers                    = server.initial_dummy_paramters_list,  # structure template
        max_norm                      = 5.0,   # clip norm of client deltas
        fraction_fit                  = 1.0,   # keep identical sampling
        fraction_evaluate             = 0.0,   # we do server-side eval instead
        min_fit_clients               = 5,
        min_available_clients         = 5,
        initial_parameters            = make_initial_parameters(server),
        evaluate_fn                   = make_evaluate_fn(server, batch_size=1),
        on_fit_config_fn              = make_on_fit_config_fn(batch_size=train_batch_size, epochs=5),
        on_evaluate_config_fn         = make_on_evaluate_config_fn(val_batch_size=1),  # <-- here
        fit_metrics_aggregation_fn    = weighted_mean,
        evaluate_metrics_aggregation_fn = weighted_mean,
    )

    # ----------------------- run the simulation -----------------------------
    fl.simulation.start_simulation(
        client_fn        = client_fn,
        num_clients      = 5,
        config           = fl.server.ServerConfig(num_rounds=15),
        strategy         = strategy,
        client_resources = {"num_cpus": 6},
        ray_init_args    = {"_system_config": {"worker_register_timeout_seconds": 120}},
    )

    log_print("Training is done", context="MAIN")



