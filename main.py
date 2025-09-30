import flwr as fl
import sys
import os
import ray
import time
import uuid
_temp_dir = f"/scratch/ssafa013/{os.environ.get('SLURM_JOB_ID', str(time.time()))}/ray/"
namespace = f"flower_ns_{uuid.uuid4().hex[:8]}"
os.environ["NUMEXPR_MAX_THREADS"] = "150"

from utils import log_print
sys.path.append('/rhome/ssafa013/DGDDPM/DGDDPM/wilds')
log_print(sys.path)
# print("salam ", sy.TYPE_REGISTRY)  # shows all registered Syft objects
from DataLoader import OpenBHBDataset
from  server import Server
from server_helper import ServerDomainSpecHelper
from client import FedClient, FlowerClientWrapper
from federated_data_splitter import FederatedOpenBHBSplitter
from strategy_custom import FedCustomStrategy
from flwr.server.strategy import FedAvg
import torch
from client_factory import ClientFactory, ValidationClientFactory
from pathlib import Path
from typing import Dict, List, Tuple, Any
# -----------------------------------------------------------------------------
import argparse, yaml
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Parameters
from utils import (debug_function, log_print,                         # your logger
                   flatten_layer_param_list_for_flower,
                   flatten_layer_param_list_for_model,
                   reconstruct_layer_param_list)
from model import BrainCancer            # your model class
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--client_cfg", required=True)   # <-- new arg
args = parser.parse_args()
with open(args.client_cfg) as f:
    CLIENT_CFG = yaml.safe_load(f)
# -----------------------------------------------------------------------------

train_batch_size=2
num_domains = 5
NUM_ROUNDS = 15

# Use a larger number of threads, say 16–64, or even all 150 cores
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
os.environ["NUMEXPR_MAX_THREADS"] = "64"
torch.set_num_threads(16)


def save_flower_weights(strategy: fl.server.strategy.Strategy,
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


# def client_fn(cid, paths, server, clients_dataset):
#     """
#     Create a FlowerClient for the given client ID and path.
#     """
#     # Create a unique directory for each client
#     cid = int(cid)
#     client_log_path = paths[cid]
#     os.makedirs(client_log_path, exist_ok=True)

#     fed_client = FedClient(client_id=cid,
#                              param_struct=server.initial_dummy_paramters,
#                              batch_size=train_batch_size)
#     fed_client.init_model(clients_dataset[cid])
#     return FlowerClientWrapper(fed_client, log_path=client_log_path)



# if __name__=='__main__':
#     # ray.init(num_cpus=90,
#     #         _temp_dir= _temp_dir,
#     #         namespace= namespace,
#     #         include_dashboard=False,
#     #         ignore_reinit_error=True )  # ← Choose enough for your number of clients
#     print("Ray session dir:", ray._private.worker._global_node._session_dir)
#     splitter = FederatedOpenBHBSplitter()
#     # ds = OpenBHBDataset()
#     clients_dataset, val_dataset, test_dataset = splitter.get_federated_splits()
#     run_id = time.strftime("%Y-%m-%d_%H-%M-%S")  # Timestamp-based run ID
    
#     server = Server(num_domains, val_dataset, test_dataset, model_type="Normal", run_id=run_id, num_rounds=NUM_ROUNDS)
#     log_base_path = server.client_log_file_paths

#     # Create Federated Strategy
#     strategy = FedCustomStrategy(server_obj=server, num_layers=server.num_layers)
#     # strategy = FedAvg()

#     client_fn = ClientFactory(log_base_path, server, clients_dataset,
#                               model_type="Normal",
#                               batch_size=train_batch_size,
#                               epochs = 5)
#     # Create a Flower client for each domain
#     # log_print("[DEBUG] About to start simulation")
#     # log_print(f"[DEBUG] num_clients: {num_domains}")
#     # log_print(f"[DEBUG] client dataset size: {len(clients_dataset)}")
#     # log_print("[DEBUG] Ray initialized with available resources:")
#     # log_print(ray.cluster_resources())
#     fl.simulation.start_simulation(
#         client_fn=client_fn,
#         num_clients=num_domains,
#         config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
#         strategy=strategy,
#         client_resources={"num_cpus": 16},
#         ray_init_args={"_system_config": {"worker_register_timeout_seconds": 1100}}
#     )
#     # federated_training(server, clients, num_layers=6, rounds=5)
#     log_print("Training is done", context="MAIN")

    

def run_federated_once(repeat_idx: int,
                       global_run_id: str) -> float:
    """A very thin wrapper around the original main-block code.
       Returns the last-round MAE."""
    # --------------- (BEGIN: copy/paste original body) -------------------
    # _temp_dir = f"/scratch/ssafa013/{os.environ.get('SLURM_JOB_ID', str(time.time()))}/ray/"
    bigdata_root = "/rhome/ssafa013/bigdata/raytmp"   # make sure this exists

    _temp_dir = f"{bigdata_root}/{os.environ.get('SLURM_JOB_ID', str(time.time()))}/ray"

    namespace = f"flower_ns_{uuid.uuid4().hex[:8]}"
    ray.init(num_cpus=5,
            num_gpus=2,          # ⚠️ new line – make 2 GPUs visible
            _temp_dir= _temp_dir,
            namespace= namespace,
            include_dashboard=False,
            ignore_reinit_error=True )  # ← Choose enough for your number of clients
    # ----- identical data split, logging paths, etc. ------------------------
    
    ray_init_args = {
        "_temp_dir": _temp_dir,
        "num_cpus": 6,
        "num_gpus": 2,
        "include_dashboard": False,
        "ignore_reinit_error": True,
        "_system_config": {"worker_register_timeout_seconds": 1100}
    }    
    splitter  = FederatedOpenBHBSplitter()
    clients_ds, val_ds, test_ds = splitter.get_federated_splits()
    
    run_id    = f"{global_run_id}/rep{repeat_idx+1}"
    server    = Server(num_clients=num_domains,
                       val_dataset=val_ds,
                       test_dataset=test_ds,
                       model_type="Normal",
                       run_id=run_id,
                       num_rounds=NUM_ROUNDS)
    # server = Server(num_domains,
    #                 val_dataset,
    #                 test_dataset,
    #                 model_type="Normal",
    #                 run_id=run_id,
    #                 num_rounds=NUM_ROUNDS)

    strategy = FedCustomStrategy(server_obj=server, num_layers=server.num_layers, client_cfg=CLIENT_CFG)

    client_fn = ClientFactory(server.client_log_file_paths,
                              server,
                              clients_ds,
                              model_type="Normal",
                              batch_size=CLIENT_CFG.get("batch_size", train_batch_size),
                              epochs=CLIENT_CFG.get("epochs", 5),
                              client_cfg=CLIENT_CFG)



    # ----------------------- run the simulation -----------------------------
    history = fl.simulation.start_simulation(
        client_fn        = client_fn,
        num_clients      = 5,
        config           = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy         = strategy,
        client_resources = {"num_cpus": 1, "num_gpus": 0.3},
        ray_init_args    = ray_init_args,
    )

    log_print("Training is done", context="MAIN")

    out_dir = Path("./runs")/f"run_{global_run_id}"
    ckpt = out_dir/f"model_run{repeat_idx+1}.pth"
    save_flower_weights(strategy, server.initial_dummy_paramters_list, ckpt)
    
    
    last_rnd   = max(history.metrics_centralized.keys())
    mae     = history.metrics_centralized['mae'][-1][1]
    loss = history.metrics_centralized['loss'][-1][1]
    spec_dist = history.metrics_centralized['spec_dist'][-1][1]
    sensitivity = history.metrics_centralized['sensitivity'][-1][1]
    ray.shutdown()
    
    return loss, mae, 0

    

    
    
    

if __name__ == "__main__":
    # # _temp_dir = f"/scratch/ssafa013/{os.environ.get('SLURM_JOB_ID', str(time.time()))}/ray/"
    # bigdata_root = "/rhome/ssafa013/bigdata/raytmp"   # make sure this exists

    # _temp_dir = f"{bigdata_root}/{os.environ.get('SLURM_JOB_ID', str(time.time()))}/ray"

    # namespace = f"flower_ns_{uuid.uuid4().hex[:8]}"
    # ray.init(num_cpus=5,
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
    #                    model_type="DANN3D",
    #                    run_id=run_id,
    #                    num_rounds=NUM_ROUNDS)

    # client_fn = ClientFactory(server.client_log_file_paths,
    #                           server,
    #                           clients_ds,
    #                           model_type="DANN3D",
    #                           batch_size=train_batch_size,
    #                           epochs=5)

    # # ------------- FedAvg, but with your hooks ------------------------------
    # strategy = TrustFedAvg(
    #     max_norm                      = 5.0,   # clip norm of client deltas
    #     fraction_fit                  = 1.0,   # keep identical sampling
    #     fraction_evaluate             = 0.0,   # we do server-side eval instead
    #     min_fit_clients               = 5,
    #     min_available_clients         = 5,
    #     initial_parameters            = make_initial_parameters(server),
    #     evaluate_fn                   = make_evaluate_fn(server, batch_size=1),
    #     on_fit_config_fn              = make_on_fit_config_fn(batch_size=train_batch_size, epochs=5),
    #     on_evaluate_config_fn         = make_on_evaluate_config_fn(val_batch_size=1),  # <-- here
    #     fit_metrics_aggregation_fn    = weighted_mean,
    #     evaluate_metrics_aggregation_fn = weighted_mean,
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


