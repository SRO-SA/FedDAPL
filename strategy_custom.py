import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import Strategy
from flwr.common import Parameters, FitRes, FitIns, Scalar
from typing import List, Tuple, Dict, Optional, Union
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from server import Server
from server_helper import ServerDomainSpecHelper
from utils import (
    get_layer_params_list, get_layer_params_dict,
    flatten_layer_param_list_for_flower, flatten_layer_param_list_for_model,
    reconstruct_layer_from_flat, reconstruct_layer_param_list,
    debug_function, log_print
)
from client_factory import ValidationClientFactory
import numpy as np
# at the top of files that use them
from utils import (
    gradE_ratio_for_layers,
    robust_mae_under_E,
    robustness_curve_auc,
    robustness_curve_auc_lite,
    directional_sensitivity_along_basis,
    basis_explained_energy,
    site_spread_stats,
)
from model import BrainCancer
import base64, zlib, json
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import os
validation_batch_size=1
from eval_actor import EvalActor
from test_ray_actor_min import CudaActor

class FedCustomStrategy(Strategy):
    def __init__(self, server_obj: Server, num_layers: int, client_cfg=None):
        
        self.server_obj = server_obj
        self.num_layers = num_layers
        self.sample_client_params_dict = None
        self.latest_client_params: Dict[int, Parameters] = {}
        self.expected_clients = server_obj.num_clients
        self.client_cfg = client_cfg
        self.server_pg = placement_group([{"CPU": 1, "GPU": 1}], strategy="STRICT_PACK")
        ray.get(self.server_pg.ready())  # holds 1 GPU; clients can't take it
        self.E_choice = "pc" # "zero" | "pc" | "dirichlet" | "lowrank"
        self.final_parameters = None  # to store final parameters after aggregation
        self.eval_actor = None
        
        cr = ray.cluster_resources()
        ar = ray.available_resources()
        log_print(f"[MAIN] cluster={cr}, avail={ar}")

        NUM_CLIENTS = 5
        CLIENT_GPU = 0.2    # Option A  (use 0.3 for Option B)
        EVAL_ACTOR_GPU = 1.0  # Option A (use 0.5 for Option B)

        total_needed = EVAL_ACTOR_GPU + NUM_CLIENTS * CLIENT_GPU
        have = cr.get("GPU", 0.0)
        if total_needed > have:
            raise RuntimeError(f"GPU oversubscription: need {total_needed}, have {have}. "
                            f"Lower client GPU fraction or eval-actor GPU.")


    @debug_function(context="SERVER STRATEGY")
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        flat_params = flatten_layer_param_list_for_flower(self.server_obj.initial_dummy_paramters_list)
        return ndarrays_to_parameters(flat_params)

    @debug_function(context="SERVER STRATEGY")
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        log_print(f"\n\n[DEBUG] aggregate_fit called for round {server_round} with {len(results)} results\n\n")
        if failures:
            for i, err in enumerate(failures):
                try:
                    print(f"[aggregate_fit] failure[{i}]: {repr(err)}")
                except Exception:
                    pass
            raise Exception("Some clients failed; retrying the round.")
        if len(results) < self.expected_clients:
            print(f"[Round {server_round}] Not all clients returned. Got {len(results)} of {self.expected_clients}.")
            raise Exception("Retrying round to get full participation.")
        
        # Collect client params (layer-structured) and weights
        client_params_dict: Dict[int, List[List[torch.Tensor]]] = {}
        client_weights: Dict[int, List[Float]] = {}
        # map from "user cid in metrics" -> Flower's client.cid
        cid_metric_to_flower: Dict[int, int] = {}

        for client, fit_res in results:
            # Your clients return their own numeric id in metrics["_id"]
            cid_metric = int(fit_res.metrics["_id"])
            cid_flower = int(client.cid)
            cid_metric_to_flower[cid_metric] = cid_flower

            flat_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            param_layers_list = reconstruct_layer_param_list(flat_ndarrays, self.server_obj.initial_dummy_paramters_list)

            client_params_dict[cid_metric] = param_layers_list
            client_weights[cid_metric] = float(fit_res.metrics.get("num_samples", 1.0))

        if self.sample_client_params_dict == None:
            self.sample_client_params_dict = client_params_dict
            
        # Run one server round: this will also populate
        # self.server_obj.shared_mean_layers[ℓ] and self.server_obj.E_layer_wise[ℓ]
        updated_client_params = self.server_obj.server_round(
            client_params_dict, self.num_layers, server_round, client_weights
        )
        # Store per-client updated params for next round's configure_fit
        for cid_metric, client_layer_params in updated_client_params.items():
            flat_params = flatten_layer_param_list_for_flower(client_layer_params)
            flower_params = ndarrays_to_parameters(flat_params)
            cid_flower = cid_metric_to_flower[cid_metric]
            self.latest_client_params[cid_flower] = flower_params

        # Choose some "global" params to return to Flower (unused in your per-client flow)
        # We can return the first client's params deterministically.
        any_flower_id = sorted(self.latest_client_params.keys())[0]
        new_global_params = self.latest_client_params[any_flower_id]
        self.final_parameters = new_global_params
        return new_global_params, {}

    @debug_function(context="SERVER STRATEGY")
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]: 
        # log_print(f"\n\n[DEBUG] configure_fit called for round {server_round} with {len(client_manager.all())} clients\n\n")
        clients = list(client_manager.all().values())
        fit_config = {}
        fit_instructions: List[Tuple[ClientProxy, FitIns]] = []
        
        # --- Build the scheduler config to send to every client this round ---
        # Round 1: before we have E_layer_wise, keep scheduler OFF
        if server_round <= 1 or len(self.server_obj.E_layer_wise) == 0:
            mode = "off"
        else:
            mode = "pc"   # or "dirichlet" / "lowrank"
        zero_last = server_round <= 6

        sched_cfg = self.server_obj.export_scheduler_config(
            mode=mode,           # "pc" | "dirichlet" | "lowrank" | "off"
            aug="consistency",          # "consistency" | "sam" | "dropout" | "irm_fd"
            rank=2,              # rank for the basis / sampler
            scale_override=None,  # optional; else per-layer scale from basis
            zero_last_layer=True,
            send_basis=True,
            send_bank=False
        )

        sched_cfg["current_round"] = server_round
        sched_cfg["num_rounds"] = self.server_obj.num_rounds
        # Add safety cap:
        sched_cfg["E_max_layer_norm"] = 0.02
        # still in configure_fit
        if server_round <= 4:
            sched_cfg["E_lambda_cons"] = 0.15
        elif server_round <= 8:
            sched_cfg["E_lambda_cons"] = 0.30
        else:
            sched_cfg["E_lambda_cons"] = 0.40
        log_print(f"sched_cfg keys: {list(sched_cfg.keys())}", context="SERVER STRATEGY")
        if "E_basis" in sched_cfg:
            log_print(f"E_basis layers: {sched_cfg['E_basis']}...", context="SERVER STRATEGY")

        # --- Per-client FitIns with personalized params and shared scheduler config ---
        if server_round == 1:
            # send the initial dummy weights to bootstrap clients
            initial_flattened_params = flatten_layer_param_list_for_flower(
                self.server_obj.initial_dummy_paramters_list
            )
            initial_flower_params = ndarrays_to_parameters(initial_flattened_params)
            for client in clients:
                fit_ins = FitIns(initial_flower_params, sched_cfg)
                fit_instructions.append((client, fit_ins))
            return fit_instructions

        # From round 2 on, we expect to have personalized params cached
        for client in clients:
            cid_flower = int(client.cid)
            if cid_flower not in self.latest_client_params:
                raise ValueError(f"[CONFIGURE_FIT] Missing parameters for client {cid_flower}")
            fit_ins = FitIns(self.latest_client_params[cid_flower], sched_cfg)
            fit_instructions.append((client, fit_ins))

        return fit_instructions

    @debug_function(context="SERVER STRATEGY")
    def configure_evaluate(self, *args, **kwargs):
        return []

    @debug_function(context="SERVER STRATEGY")
    def aggregate_evaluate(self, server_round, results, failures):
        return super().aggregate_evaluate(server_round, results, failures)
    
    @debug_function(context="SERVER STRATEGY EVALUATION")
    def evaluate(self, server_round, parameters):
        #Start Validation
        log_print("Starting Validation", context="SERVER STRATEGY EVALUATION")
        # Skip early evaluation before M_spec is available
        if not self.server_obj.E_layer_wise or not self.server_obj.shared_mean_layers:
            log_print(f"[WARNING] Skipping evaluation at round {server_round} — residual bank not initialized yet.", context="EVALUATE")
            return None

        validation_log_path = self.server_obj.validation_log_path
        server_helper = ServerDomainSpecHelper(self.server_obj)
        # val_cleint_factory = ValidationClientFactory(self.server_obj, self.server_obj.val_dataset,
        #                                             model_type="Normal",
        #                                             batch_size=validation_batch_size,
        #                                             client_cfg=self.client_cfg)
        sched_cfg = self.server_obj.export_scheduler_config(
                mode="off",           # "pc" | "dirichlet" | "lowrank" | "off"
                aug="consistency",          # "consistency" | "sam" | "dropout" | "irm_fd"
                rank=2,              # rank for the basis / sampler
                scale_override=0.8,  # optional; else per-layer scale from basis
                zero_last_layer=True,
                send_basis=True,
                send_bank=False
            )
        sched_cfg["current_round"] = server_round
        sched_cfg["num_rounds"] = self.server_obj.num_rounds
        # ---- Sample ONE residual per layer (PC sampler on residual bank) ----
        # You can try: sample_residual_dirichlet / sample_residual_lowrank_mahalanobis
        E_choice_layers = {}
        if self.E_choice == "zero":
            # Use zeros *matching the shared mean* (flat vector per layer)
            for l in range(self.server_obj.num_layers):
                shared_flat = self.server_obj.shared_mean_layers.get(l)
                if shared_flat is None:
                    # Fallback: derive d from reference layer structure
                    ref = self.server_obj.initial_dummy_paramters_list[l]
                    d_l = sum(p.numel() for p in ref)
                    device = ref[0].device if len(ref) > 0 else torch.device("cpu")
                    dtype  = ref[0].dtype  if len(ref) > 0 else torch.float32
                    E_choice_layers[l] = torch.zeros(d_l, device=device, dtype=dtype)
                else:
                    E_choice_layers[l] = torch.zeros_like(shared_flat)

        elif self.E_choice == "pc":
            E_choice_layers = server_helper.sample_residual_pc(
                self.server_obj.E_layer_wise, frac=0.4
            )
        elif self.E_choice == "dirichlet":
            E_choice_layers = server_helper.sample_residual_dirichlet(
                self.server_obj.E_layer_wise, alpha=0.3
            )
        elif self.E_choice == "lowrank":
            E_choice_layers = server_helper.sample_residual_lowrank_mahalanobis(
                self.server_obj.E_layer_wise, epsilon=1.0
            )
        else:
            raise ValueError(f"Unknown E_choice: {self.E_choice}. Choose from 'zero', 'pc', 'dirichlet', 'lowrank'.")

        # Enforce zero residual on the output head during validation
        if sched_cfg.get("E_zero_last_layer", True):
            last = self.server_obj.output_layer_idx
            E_choice_layers[last] = torch.zeros_like(self.server_obj.shared_mean_layers[last])

        # Ensure every layer ℓ has a correctly shaped residual
        for l, shared_flat in self.server_obj.shared_mean_layers.items():
            if l not in E_choice_layers or E_choice_layers[l].shape != shared_flat.shape:
                E_choice_layers[l] = torch.zeros_like(shared_flat)

        # ---- Evaluate shared_mean + sampled residual on validation ----
        validation_logs, sens, spec_dist, val_client = self._eval_E_on_val(
            self.server_obj, server_helper, E_choice_layers, sched_cfg = sched_cfg,
            batch_size=validation_batch_size
        )

        
        loader = val_client.local_dataloader
        
        cached = []
        for i, (x, y, _m) in enumerate(loader):
            if i >= 2: break                      # 2 batches is plenty
            maxB = min(8, x.shape[0])
            xb = x[:maxB].contiguous().cpu().numpy()
            yb = y[:maxB].contiguous().cpu().numpy()
            cached.append((x[:maxB].contiguous().cpu().numpy(),
                        y[:maxB].contiguous().cpu().numpy()))


        x_np = xb; y_np = yb
        # in strategy.evaluate():
        state_dict_cpu = {k: v.detach().cpu().numpy() for k, v in val_client.model.state_dict().items()}
        e_basis_b64 = sched_cfg.get("E_basis", None)
        
        act = self._ensure_eval_actor()  # server eval on CPU

        robust_res = {"val_mae_batch": 0.0, "robustness_proxy": 0.0}
        if xb is not None:
            robust_res_ref = act.run.remote(
                state_dict_cpu,
                e_basis_b64,
                x_np, y_np,
                out_idx=self.server_obj.output_layer_idx,
                zero_out_last_layer=sched_cfg.get("E_zero_last_layer", True),
            )
            robust_res = ray.get(robust_res_ref, timeout=120.0)


        # ---- Build a torch basis from what we just sent in sched_cfg ----
        basis = {}
        try:
            if "E_basis" in sched_cfg:  # sched_cfg has packed b64 string
                blob = base64.b64decode(sched_cfg["E_basis"])
                data = json.loads(zlib.decompress(blob).decode("utf-8"))
                for k_str, entry in data.items():
                    shape = tuple(entry["shape"])
                    dt    = np.float16 if entry["dtype"] == "float16" else np.float32
                    U_np  = np.frombuffer(base64.b64decode(entry["u_b64"]), dtype=dt).reshape(shape).astype(np.float32)
                    basis[int(k_str)] = {"U": torch.from_numpy(U_np), "scale": float(entry["scale"])}
        except Exception as e:
            log_print(f"[EVAL] basis decode failed: {e}", context="EVAL")

        # ---- Robustness under E-perturbations ----
        try:
            mae_clean, mae_E_mean, mae_E_worst = robust_mae_under_E(
                model=val_client.model,
                dataloader=val_client.local_dataloader,
                basis=basis,
                get_params_for_layers_fn=val_client.model.get_params_for_layers,
                reconstruct_from_flat_fn=reconstruct_layer_from_flat,
                m=5,                      # 5 trials per round
                step_scale=None,          # use per-layer scale from basis
                zero_out_last_layer=True,
                out_idx=self.server_obj.output_layer_idx,
            )
            # mae_by_scale, auc_mae = robustness_curve_auc_lite(
            #     model=val_client.model,
            #     dataloader=val_client.local_dataloader,
            #     basis=basis,
            #     get_params_for_layers_fn=val_client.model.get_params_for_layers,
            #     reconstruct_from_flat_fn=reconstruct_layer_from_flat,
            #     scales=[0.0, 0.2, 0.4, 0.6, 0.8],
            #     trials_per_scale=2,
            #     zero_out_last_layer=True,
            #     out_idx=self.server_obj.output_layer_idx,
            # )
            # mae_by_scale, auc_mae = robustness_curve_auc_lite(
            #     model=val_client.model,
            #     dataloader=val_client.local_dataloader,
            #     basis=basis,  # or self.server_obj.E_basis decoded
            #     get_params_for_layers_fn=val_client.model.get_params_for_layers,
            #     reconstruct_from_flat_fn=reconstruct_layer_from_flat,
            #     scales=(0.0, 0.4, 0.8),
            #     trials_per_scale=1,
            #     max_batches=1,             # ← the big speedup
            #     use_top1_only=True,
            #     zero_out_last_layer=True,
            #     out_idx=self.server_obj.output_layer_idx,
            # )
            auc_ref = act.auc_on_cached_batches.remote(
                state_dict_cpu,
                e_basis_b64,
                cached,
                out_idx=self.server_obj.output_layer_idx,
                zero_out_last_layer=sched_cfg.get("E_zero_last_layer", True),
                scales=[0.0, 0.4, 0.8],             # fewer points for speed
                trials_per_scale=1,                 # 1 trial per scale
            )
            auc_res = ray.get(auc_ref, timeout=120.0)  # {"mae_by_scale": {...}, "auc": ...}
            mae_by_scale = auc_res.get("mae_by_scale", {})
            auc_mae = auc_res.get("auc", 0.0)
        except Exception as e:
            mae_clean = mae_E_mean = mae_E_worst = 0.0
            mae_by_scale = {}
            auc_mae = 0.0
            log_print(f"[EVAL] robust metrics failed: {e}", context="EVAL")

        # ---- Gradient projection ratio on validation (one batch) ----
        try:
            _perL, gradE_mean, gradE_p95 = gradE_ratio_for_layers(
                model=val_client.model,
                dataloader=val_client.local_dataloader,
                basis=basis,
                get_params_for_layers_fn=val_client.model.get_params_for_layers,
                n_batches=1,
                zero_out_last_layer=True,
                out_idx=self.server_obj.output_layer_idx,
            )
        except Exception as e:
            gradE_mean = gradE_p95 = 0.0
            log_print(f"[EVAL] gradE_ratio failed: {e}", context="EVAL")

        # ---- Optional: directional sensitivity along first basis vector ----
        try:
            dir_sens = directional_sensitivity_along_basis(
                model=val_client.model,
                dataloader=val_client.local_dataloader,
                basis=basis,
                get_params_for_layers_fn=val_client.model.get_params_for_layers,
                reconstruct_from_flat_fn=reconstruct_layer_from_flat,
                eps=1e-3,
                n_batches=1,
                zero_out_last_layer=True,
                out_idx=self.server_obj.output_layer_idx,
            )
        except Exception as e:
            dir_sens = {}
            log_print(f"[EVAL] directional_sensitivity failed: {e}", context="EVAL")


        # ---- Logging ----
        with open(validation_log_path, 'a') as f:
            f.write(f"Server Round: {server_round}\n")
            f.write(f"‖E_sample − 0‖₂ (global flat) = {spec_dist:.2f}\n")
            f.write(f"E-robust: MAE_clean={mae_clean:.4f}, MAE_E_mean={mae_E_mean:.4f}, MAE_E_worst={mae_E_worst:.4f}\n")
            f.write(f"E-robust: AUC_MAE={auc_mae:.4f}, MAE_by_scale={json.dumps(mae_by_scale)}\n")
            f.write(f"E-grad: mean_ratio={gradE_mean:.4f}, p95_ratio={gradE_p95:.4f}\n")
            f.write(f"robust_proxy: {robust_res['robustness_proxy']:.4f}, "
                    f"val_mae_batch: {robust_res['val_mae_batch']:.4f}\n")
            if dir_sens:
                # keep it short: show a few layers
                top = sorted(dir_sens.items())[:3]
                f.write(f"E-dir_sens(sample)={top}\n")
            #TODO Update Later to include Site MSE and Macro MSE
            for epoch, loss, mae, site_mse, macro_mse, var_site, gap9010 in validation_logs:
                f.write(f"E-sample: Loss={loss:.4f}, MAE={mae:.4f}, Macro_MSE={macro_mse:.4f}, var={var_site:.4f}, gap90-10={gap9010:.4f}\n")

        with open(self.server_obj.sensitivity_log_path, 'a') as g:
            g.write(f"{server_round},E_sample," + ",".join(f"{x:.4f}" for x in sens) + "\n")

        super().evaluate(server_round, parameters)
        loss = validation_logs[0][1]  # First loss value
        return float(loss), {
            "loss": validation_logs[0][1],  # First loss value
            "mae": validation_logs[0][2],    # First MAE value
            "spec_dist": spec_dist,           # Distance of E from 0
            "sensitivity": sens,              # Per-layer sensitivity
        }


    # ------------------------------------------------------------
    #  helper: build full network (inv + flat_spec) and compute
    #          loss/mae on the global validation set
    # ------------------------------------------------------------
    # def _eval_spec_on_val(self, server, server_helper, flat_spec, batch_size=1):
    #     ref_layers  = server.initial_dummy_paramters_list
    #     split_sizes = [p.numel() for layer in ref_layers for p in layer]
    #     flat_parts  = list(torch.split(flat_spec, split_sizes))

    #     full_layers = []
    #     idx = 0
    #     for l, ref in enumerate(ref_layers):
    #         spec_l_flat = torch.cat(flat_parts[idx : idx + len(ref)])
    #         idx += len(ref)
    #         full_flat  = server.inv_agg[l] + spec_l_flat
    #         full_layer = reconstruct_layer_from_flat(full_flat, ref)
    #         full_layers.append(full_layer)

    #     val_factory = ValidationClientFactory(
    #         server, server.val_dataset, model_type="Normal", batch_size=batch_size
    #     )
    #     val_client = val_factory(0, full_layers)
        
    #     # after you create val_client
    #     model = val_client.model
    #     # layers_flat = []
    #     # for lp in model.get_params_for_layers():        # list[list[tensor]]
    #     #     flat = torch.cat([p.view(-1) for p in lp])
    #     #     flat.requires_grad_(True)                   # share storage
    #     #     flat.retain_grad()          # <── keep .grad for non-leaf
    #     #     layers_flat.append(flat)
    #     layer_param_groups = get_layer_params_list(model, clone=False)
    #     sens = server_helper.layer_grad_norms(
    #                             model,
    #                             val_client.local_dataloader,
    #                             layer_param_groups,
    #                             n_batches=1)
    #     # sens is list of length num_layers
    #     validation_logs = val_client.local_train_step(is_train=False)
    #     return validation_logs, sens
    
    # ------------------------------------------------------------
    # NEW helper: build full network (shared + E_choice) and compute
    #             loss/mae on the global validation set
    # ------------------------------------------------------------
    def _eval_E_on_val(self, server: Server, server_helper: ServerDomainSpecHelper,
                       E_choice_layers: Dict[int, torch.Tensor], sched_cfg: Dict[str, Optional[Union[str, int]]], batch_size=1):
        """
        Compose θ_val by θ_val[ℓ] = shared_mean_layers[ℓ] + E_choice_layers[ℓ].
        Evaluate on the validation set; also compute per-layer grad-norm sensitivity.
        """
        ref_layers = server.initial_dummy_paramters_list
        full_layers = []
        # also build a single global flat E for logging distance (optional)
        flat_E_all = []

        for l, ref in enumerate(ref_layers):
            shared_flat = server.shared_mean_layers[l]                 # [d_l]
            e_flat = E_choice_layers.get(l, torch.zeros_like(shared_flat))
            full_flat = shared_flat + e_flat
            layer_struct = reconstruct_layer_from_flat(full_flat, ref)
            full_layers.append(layer_struct)
            flat_E_all.append(e_flat)

        # create a validation client with these parameters

        val_client_cfg = self.client_cfg.copy()
        val_client_cfg['device'] = 'cpu'
        val_factory = ValidationClientFactory(
            server, server.val_dataset, model_type="Normal", batch_size=batch_size, 
            client_cfg=val_client_cfg
        )
        # --- force CPU for server-side validation ---
        # val_client.device = torch.device("cpu")
        # val_client.model = val_client.model.to("cpu")
        
        val_client = val_factory(0, full_layers)

        val_client.device_pref = "cpu"
        val_client.model = val_client.model.to("cpu")
        # torch.backends.cudnn.enabled = False  # optional: ensure no CUDA/cuDNN path is tried

        # Sensitivity (per-layer grad norms on a few batches)
        model = val_client.model
        layer_param_groups = get_layer_params_list(model, clone=False)
        sens = server_helper.layer_grad_norms(
            model, val_client.local_dataloader, layer_param_groups, n_batches=1
        )

        # Standard evaluation
        validation_logs = val_client.local_train_step(is_train=False, config=sched_cfg)

        # Ensure any GPU work from eval is done before we measure spec_dist
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # E_choice_layers was used to build flat_E_all earlier; use it directly
        # Make sure we have a list/dict of 1-D tensors
        E_list = list(E_choice_layers.values()) if isinstance(E_choice_layers, dict) else flat_E_all

        spec_dist_sq = 0.0
        for v in E_list:
            # move chunk-by-chunk to CPU to avoid a huge cat and to make sync explicit
            vv = v.detach().to("cpu", dtype=torch.float32, copy=True).view(-1)
            spec_dist_sq += float((vv * vv).sum().item())

        spec_dist = spec_dist_sq ** 0.5

        return validation_logs, sens, spec_dist, val_client


    @debug_function(context="SERVER STRATEGY EVAL ACTOR")
    def _ensure_eval_actor(self):
        
        pg = placement_group([{"CPU": 1, "GPU": 1}], strategy="STRICT_PACK")
        ray.get(pg.ready())

        a = CudaActor.options(
            num_cpus=1, num_gpus=0.2,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=0, placement_group_capture_child_tasks=True
            ),
        ).remote(True)
        ray.get(a.ping.remote(), timeout=60)
        log_print("Basic Actor Done...", context="SERVER STRATEGY EVAL ACTOR")

        act = getattr(self, "eval_actor", None)
        if act is not None:
            try:
                ray.get(act.ping.remote(), timeout=500.0)
                return act
            except Exception:
                self.eval_actor = None

        client_cfg = self.client_cfg or {}
        
        log_print("Creating EvalActor...", context="SERVER STRATEGY EVAL ACTOR")
        h = EvalActor.options(
            num_cpus=1,
            num_gpus=0.2,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.server_pg,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            ),
            max_restarts=1,
            max_task_retries=1).remote(client_cfg, True)
        log_print("EvalActor handle obtained, pinging...", context="SERVER STRATEGY EVAL ACTOR")
        ray.get(h.ping.remote(), timeout=120.0)
        log_print("EvalActor responsive.", context="SERVER STRATEGY EVAL ACTOR")
        self.eval_actor = EvalActor.options(
            num_cpus=1,
            num_gpus=0.2,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.server_pg,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            ),
            max_restarts=1,
            max_task_retries=1,
        ).remote(client_cfg, True)  # (client_cfg, use_cuda=True)
        ray.get(self.eval_actor.ping.remote(), timeout=500.0)
        return self.eval_actor