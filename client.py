from calendar import c
from math import log
from model import BrainCancer
from model_feature_regress import DANN3D
import torch
import logging, os, pathlib
import ray
from torch import log_, nn
import numpy as np
import os
import sys
import base64, json, zlib

sys.path.append('/rhome/ssafa013/DGDDPM/DGDDPM/wilds')
from time import time
from torch import optim
from tensorboardX import SummaryWriter
import math
from typing import Optional, Dict, Any
from collections import defaultdict
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from utils import debug_function, log_print, flatten_layer_param_list_for_model, flatten_layer_param_list_for_flower
from utils import PerRoundEScheduler, reconstruct_layer_from_flat

# at the top of files that use them
from utils import (
    gradE_ratio_for_layers,
    robust_mae_under_E,
    robustness_curve_auc,
    directional_sensitivity_along_basis,
    basis_explained_energy,
    site_spread_stats,
)

TOTAL_DATASET_SIZE = 1587
site_counts = [0, 24, 277, 43, 25, 47, 50, 17, 11, 14, 10, 73, 956, 20, 20]

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_MAX_THREADS"] = "4"

def get_num_cpus():
    try:
        # If inside a Ray actor context
            ctx = ray.get_runtime_context()
            cpu_resources = ctx.get_assigned_resources().get("CPU", 1)
            return int(cpu_resources)
    except Exception:
        pass

    # Fallback: use total system CPUs
    return os.cpu_count()


class FedClient():
    def __init__(self, client_id: str, param_struct: list = None, batch_size: int=1, lr: float=7.5e-4, epochs = 10, cfg: Optional[Dict[str, Any]] = None):
        """
        client_id: unique identifier
        local_data: local dataset (details omitted)
        param_struct: list of layer tensors, e.g. [layer0, layer1, ...]
                      each layer is a torch.Tensor (potentially flattened).
        """
        # num_cpus = get_num_cpus()
        logging.getLogger(f"client.{client_id}").log(getattr(logging, "DEBUG", logging.DEBUG), "Avaialbe cpus are {num_cpus}\n")
        torch.set_num_threads(4)
    # Split Train dataset by Domain number
        self.client_id = client_id
        self.param_struct = param_struct  # list of Tensors
        self.num_layers = len(param_struct) if param_struct is not None else 0
        self.batch_size = batch_size
        # Initialize after send
        self.epochs = epochs
        self.local_data = None
        self.model = None
        self.opt = None
        self.opt_disc = None
        self.cfg = cfg
        self.lr = self.cfg.get("lr", lr)
        self.d_lr = self.cfg.get("d_lr", self.lr * 2.0)
        # --------- SCAFFOLD stuff (lazy-initialised) -----------
        self.scaffold_active    = False     # auto-detect mode
        self.c_global_disc      = None      # list[np.ndarray]
        self.ci_disc            = None      # list[np.ndarray]
        self.n_ref_layers = None   # <- added
        self.standard_dt_size = 78
        self.max_dt_size = 995
        self.E_sched: Optional[PerRoundEScheduler] = None
        self.output_layer_idx =  0  # adjust if known (e.g., 4)

        self.device_pref = (self.cfg or {}).get("device", "auto")  # "auto"|"cpu"|"cuda"
        # in FedClient.__init__
        self.use_E_consistency = False
        self.lambda_cons = 0.25

        self.use_E_SAM = False
        self.sam_eps = 5e-3

        self.use_E_dropout = False
        self.dropout_keep = 0.8

        self.use_IRM_FD = False
        self.lambda_irm = 0.2

    def _resolve_device(self):
        if self.device_pref == "cpu":
            return torch.device("cpu")
        if self.device_pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_layer_flats(self):
        """
        Returns list[Tensor] where each tensor is a view into the model
        weights of one layer.  No extra memory is allocated.
        """
        flat_list = []
        for layer_params in self.model.get_params_for_layers():  # list[list[tensor]]
            flat = torch.cat([p.view(-1) for p in layer_params])
            flat.requires_grad_(True)        # ensure grad will be filled
            flat_list.append(flat)
        return flat_list
    
    @debug_function(context="CLIENT")
    def init_model(self, local_data, model, is_train=True):
        self.data_size = len(local_data)
        self.UseAdam = self.cfg.get("UseAdam", True)
        # Bmax = self.max_dt_size                     # largest site size
        # Bmin = self.standard_dt_size               # smallest site size
        # Bcur = self.data_size
        # alpha = 0.5      
        # scale = max(0.2, min(1.0, math.sqrt(Bcur / Bmax)))
        # self.d_lr = self.lr * scale
        # self.d_lr = self.lr * 2.0
        self.device = self._resolve_device()
        logging.getLogger(f"client.{self.client_id}").log(getattr(logging, "DEBUG", logging.DEBUG), f"Normal CLIENT {self.client_id} Device is {self.device}\n")

        # if torch.cuda.is_available():
        #     self.model = model.cuda() #BrainCancer().cuda()
        # else:
        #     self.model = model #BrainCancer()
        self.model = model.to(self.device)
        if isinstance(self.model, BrainCancer):
            if self.UseAdam:
                self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
                self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=0.5)

            else:
                self.opt = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
                self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=5, gamma=0.5)

        elif isinstance(self.model, DANN3D):
            self.ProxAll = self.cfg.get("ProxAll", False)
            if self.UseAdam:
                self.opt = optim.Adam(list(self.model.featurizer.parameters()) + list(self.model.regressor.parameters()), lr=self.lr)
                self.opt_disc = optim.Adam(self.model.domain_classifier.parameters(), lr=self.d_lr)
                self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=0.5)

            else:
                self.opt = optim.SGD(list(self.model.featurizer.parameters()) + list(self.model.regressor.parameters()), lr=self.lr, momentum=0.9, weight_decay=1e-4)
                self.opt_disc = optim.SGD(self.model.domain_classifier.parameters(), lr=self.d_lr, momentum=0.9, weight_decay=1e-4)    
                self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=5, gamma=0.5)

        # if self.param_struct is not None:
        self.model.receive_and_update_params(flatten_layer_param_list_for_model(self.param_struct))
        if self.n_ref_layers is None:
            # the model is now materialised, we can count safely
            flat = flatten_layer_param_list_for_flower(self.model.get_params_for_layers())
            self.n_ref_layers = len(flat)                                 # 14 (tensors)
        self.local_data = local_data
        if(is_train):
            self.weight = self.data_size/TOTAL_DATASET_SIZE
            self.local_dataloader = get_train_loader("standard", self.local_data, batch_size=self.batch_size, drop_last=True, num_workers=1)
        else:
            self.weight = 1
            self.local_dataloader = get_eval_loader("standard", self.local_data, batch_size=self.batch_size, drop_last=True, num_workers=1)
        
        self.output_layer_idx =  self.num_layers - 1  # adjust if known (e.g., 4)
    @debug_function(context="CLIENT")
    def local_train_step(self, is_train=True, config=None):
        """
        Example local training step. In real usage, you'd define a nn.Module 
        or tie param_struct to actual layer weights and do forward/backward passes.
        Here we just show a placeholder.
        """
        config = config or {}
        
        # right after: config = config or {}
        aug = str(config.get("E_client_aug", "none")).lower()
        self.use_E_consistency = (aug == "consistency") or bool(config.get("E_lambda_cons", 0) > 0)

        logging.getLogger(f"client.{self.client_id}").debug(
            f"[cfg] E_mode={config.get('E_mode')} aug={aug} "
            f"has_basis={'E_basis' in config} lambda_cons={config.get('E_lambda_cons')} "
            f"active={(self.E_sched and self.E_sched.mode)}"
        )
        # Pseudocode: 
        #   1) build or update your local model with self.param_struct
        #   2) compute gradient on local_data
        #   3) update param_struct
        # Activate/adjust scheduler from config (if provided)
        aug = config.get("E_client_aug", "none")
        self.use_E_consistency = (aug == "consistency")
        self.use_E_SAM        = (aug == "sam")
        self.use_E_dropout    = (aug == "dropout")
        self.use_IRM_FD       = (aug == "irm_fd")

        self.lambda_cons  = float(config.get("E_lambda_cons",  self.lambda_cons))
        self.sam_eps      = float(config.get("E_sam_eps",      self.sam_eps))
        self.dropout_keep = float(config.get("E_dropout_keep", self.dropout_keep))
        self.lambda_irm   = float(config.get("E_lambda_irm",   self.lambda_irm))

        if self.E_sched is not None:
            # Expecting small keys; all optional
            mode = config.get("E_mode", "off")
            self.E_sched.mode = mode
            self.E_sched.rank = int(config.get("E_rank", 1))
            self.E_sched.scale = float(config.get("E_scale", 0.4))
            self.E_sched.max_layer_norm = float(config.get("E_max_layer_norm", 0.0)) or None
            self.E_sched.zero_out_last_layer = bool(config.get("E_zero_last_layer", True))
            # Optional: residual basis or bank pushed by server (keep small!)
            # If provided as Python lists/np arrays by Flower, convert to tensors on the right device
            if "E_basis" in config:
                dev   = next(self.model.parameters()).device
                dtype = next(self.model.parameters()).dtype

                blob  = base64.b64decode(config["E_basis"])
                data  = json.loads(zlib.decompress(blob).decode("utf-8"))

                basis = {}
                for k_str, entry in data.items():
                    shape = tuple(entry["shape"])
                    dt    = np.float16 if entry["dtype"] == "float16" else np.float32
                    raw   = base64.b64decode(entry["u_b64"])
                    U_np  = np.frombuffer(raw, dtype=dt).reshape(shape).astype(np.float32)
                    U     = torch.from_numpy(U_np).to(device=dev, dtype=dtype)
                    basis[int(k_str)] = {"U": U, "scale": float(entry["scale"])}

                self.E_sched.basis = basis
                keys = sorted(basis.keys())
                logging.getLogger(f"client.{self.client_id}").debug(
                    f"[E_basis] received n_layers={len(keys)} keys={keys} "
                    f"(min={min(keys) if keys else 'NA'}, max={max(keys) if keys else 'NA'}) "
                    f"rank={self.E_sched.rank} example_shape={basis[keys[0]]['U'].shape if keys else 'NA'}"
                )
                
            if "E_bank" in config:
                bank = {}
                for k_str, arr in config["E_bank"].items():
                    k = int(k_str)
                    bank[k] = torch.tensor(arr, dtype=next(self.model.parameters()).dtype, device=next(self.model.parameters()).device)
                self.E_sched.bank = bank
            # Ensure scheduler uses the latest snapshot (in case set_parameters ran just before)
            if self.round_start_snapshot is not None:
                self.E_sched.set_round_start_snapshot(self.round_start_snapshot)
            
        logs = []  # (epoch, loss, acc) for each epoch
        if is_train:         
            if isinstance(self.model, BrainCancer):
                for epoch in range(self.epochs):
                    loss, mae = self.train_normal_model(epoch)
                    logs.append((epoch, float(loss), float(mae))) # Ensure return values are serializable
            elif isinstance(self.model, DANN3D):
                self.disc_global = [p.detach().clone() for p in self.model.domain_classifier.parameters()]
                self.feat_global = [p.detach().clone() for p in self.model.featurizer.parameters()]
                self.reg_global  = [p.detach().clone() for p in self.model.regressor.parameters()]
                for epoch in range(self.epochs):
                    loss, mae, d_loss, p_loss, ent, ent_acc = self.train_dann3d_model(epoch, config)
                    logs.append((epoch, float(loss), float(mae), float(d_loss), float(p_loss), float(ent), float(ent_acc)))
            else:
                raise ValueError("Unknown model type")
            self.sched.step()
            return logs
        else:
            if isinstance(self.model, BrainCancer):
                loss, mae, site_mse, macro_mse, var_site, gap9010 = self.evaluate_normal_model()
                logs.append((0, float(loss), float(mae), site_mse, float(macro_mse), float(var_site), float(gap9010)))
            elif isinstance(self.model, DANN3D):
                loss, mae, site_mse, macro_mse  = self.evaluate_dann3d_model(config)
                logs.append((0, float(loss), float(mae), site_mse, float(macro_mse)))
            else:
                raise ValueError("Unknown model type")
            return logs
    
    @debug_function(context="CLIENT")
    def receive_and_update_params(self, new_params_for_layers):
        """
        new_params_for_layers:list of layer tensors returned from server_round,
                               i.e. [ layer0_new,  layer1_new, ...]
        Overwrite or merge them with local param_struct as you see fit.
        Here we just do direct overwrite.
        """
        assert len(new_params_for_layers) == self.n_ref_layers
        # 1) Load server params into the model
        self.model.receive_and_update_params(new_params_for_layers)

        # 2) Snapshot round-start weights (for per-epoch restore)
        layers_now = self.model.get_params_for_layers()
        true_L = len(layers_now)
        true_out = true_L - 1
        self.round_start_snapshot = [[t.detach().clone() for t in layer] for layer in layers_now]

        # 3) (Re)build scheduler with default 'off' (it can be switched on by fit(config))
        if self.E_sched is None:
            self.E_sched = PerRoundEScheduler(
                model=self.model,
                get_params_for_layers_fn=self.model.get_params_for_layers,
                reconstruct_from_flat_fn=reconstruct_layer_from_flat,
                layer_count=true_L,
                output_layer_idx=true_out,
                mode="off",
                device=next(self.model.parameters()).device
            )
        else:
            # keep scheduler in sync if anything changes
            self.E_sched.L = true_L
            self.E_sched.out_idx = true_out

        # set snapshot into scheduler
        self.E_sched.set_round_start_snapshot(self.round_start_snapshot)
        logging.getLogger(f"client.{self.client_id}").debug(
            f"[E_sched:init] L={self.E_sched.L} out_idx={self.E_sched.out_idx} zero_last={self.E_sched.zero_out_last_layer}"
        )

    @debug_function(context="CLIENT")
    def get_params(self):
        """ Return a dict of layer parameters for the server to gather. """
        # layers = self.model.get_params_for_layers()
        # for i, layer in enumerate(layers):
        #     log_print(f"Layer {i}: {len(layer)}", context="GET PARAMS")
        return self.model.get_params_for_layers()
    
    def prox_term(self, params_iterable, globals_iterable, mu):
        return 0.5 * mu * sum((p - pg).pow(2).sum()
                            for p, pg in zip(params_iterable, globals_iterable))

# '''
#     @debug_function(context="CLIENT")
#     def train_normal_model(self, epoch): #(self, epoch, writer, log_every=100)
        
#         _ = self.model.train()
#         # ---- NEW: if scheduler active, restore snapshot and add a fresh E for this epoch
#         if self.E_sched is not None and self.E_sched.active():
#             self.E_sched.restore_snapshot()
#             delta_layers = self.E_sched.build_epoch_delta()   # dict: layer_idx -> flat tensor
#             # NEW: measure norms to ensure non-zero deltas
#             with torch.no_grad():
#                 dn = {ℓ: float(d.norm().item())
#                     for ℓ, d in delta_layers.items()
#                     if d is not None and d.numel() > 0}
#             logging.getLogger(f"client.{self.client_id}").debug(
#                 f"[E_sched] mode={self.E_sched.mode} scale={self.E_sched.scale} "
#                 f"zero_last={self.E_sched.zero_out_last_layer} delta_norms={dn}"
#             )
#             self.E_sched.apply_delta_layers(delta_layers)
#         # -------------------------------------------------------------------------------

#         # ---------- A) gradE_ratio: do it once per epoch (cheap) ----------
#         gradE_mean = None
#         gradE_p95  = None
#         if (self.E_sched is not None and self.E_sched.active() and
#             getattr(self.E_sched, "basis", None)):   # only if basis exists
#             try:
#                 per_layer, gradE_mean, gradE_p95 = gradE_ratio_for_layers(
#                     model=self.model,
#                     dataloader=self.local_dataloader,
#                     basis=self.E_sched.basis,  # dict[int]->{"U": Tensor, "scale": float}
#                     get_params_for_layers_fn=self.model.get_params_for_layers,
#                     n_batches=1,  # cheap
#                     zero_out_last_layer=self.E_sched.zero_out_last_layer,
#                     out_idx=self.output_layer_idx,
#                 )
#                 logging.getLogger(f"client.{self.client_id}").debug(
#                     f"[gradE_ratio] mean={gradE_mean:.4f} p95={gradE_p95:.4f} per_layer={per_layer}"
#                 )
#             except Exception as e:
#                 logging.getLogger(f"client.{self.client_id}").warning(
#                     f"[gradE_ratio] failed: {e}"
#                 )
#                 gradE_mean, gradE_p95 = None, None
#         # ------------------------------------------------------------------

#         y_preds = []
#         y_trues = []
#         losses = []
#         cons_pens = []
#         maes = []
#         layer_param_groups = self.model.get_params_for_layers()  # reuse across batches
#         loss_metric = nn.MSELoss()
#         mae_metric = nn.L1Loss()
        

#         for i, (image, label, metadata) in enumerate(self.local_dataloader):
            
#             # ---------- put this at the top of train_dann3d_model -------------
#             # import hashlib, numpy as np, torch

#             # def sha(flat):
#             #     v = np.concatenate([p.flatten() for p in flat])
#             #     return hashlib.sha1(v).hexdigest()
#             # -------------------------------------------------------------------
            
#             self.opt.zero_grad()
#             # log_print(f"Epoch {epoch}, Batch {i}: image shape={image.shape}, label shape={label.shape}", context="CLIENT TRAINING")
#             # if torch.cuda.is_available():
#             #     image = image.cuda()
#             #     label = label.cuda()
#                 # weight = weight.cuda()
#             image = image.to(self.device)
#             label = label.to(self.device)

#             prediction = self.model(image.float())
#             base_loss = loss_metric(prediction, label)

#             # 2) optional consistency (output-level)
#             if self.use_E_consistency and self.E_sched is not None and self.E_sched.active():
#                 flats  = [torch.cat([p.view(-1) for p in plist]) for plist in layer_param_groups]
#                 deltas = self.E_sched.sample_delta_E(flats)
#                 # LOG the sampled deltas used for consistency (not the epoch-start ones)
#                 try:
#                     dn_cons = [float(d.norm().item()) for d in deltas]
#                 except Exception:
#                     dn_cons = []
#                 logging.getLogger(f"client.{self.client_id}").debug(f"[consistency] delta_norms={dn_cons}")

#                 with self.E_sched.apply_delta_temporarily(layer_param_groups, deltas):
#                     pred_pert  = self.model(image.float())
#                 cons_pen = torch.mean((pred_pert - prediction).pow(2))
#                 base_loss = base_loss + self.lambda_cons * cons_pen
#                 cons_pens.append(cons_pen.item())

#             # 3) optional IRM-FD (loss-level)
#             if self.use_IRM_FD and self.E_sched is not None and self.E_sched.active():
#                 flats  = [torch.cat([p.view(-1) for p in plist]) for plist in layer_param_groups]
#                 deltas = self.E_sched.sample_delta_E(flats)
#                 with self.E_sched.apply_delta_temporarily(layer_param_groups, deltas):
#                     pred_eps  = self.model(image.float())
#                     loss_eps  = loss_metric(pred_eps, label)
#                 irm_pen  = torch.relu(loss_eps - base_loss.detach())
#                 base_loss = base_loss + self.lambda_irm * irm_pen

#             # 4) E-dropout mask (occasional)
#             if self.use_E_dropout and self.E_sched is not None and self.E_sched.active() and (i % 5 == 0):
#                 flats = [torch.cat([p.view(-1) for p in plist]) for plist in layer_param_groups]
#                 L = len(flats); deltas = []
#                 for ℓ, w in enumerate(flats):
#                     d, device, dtype = w.numel(), w.device, w.dtype
#                     U, r = self.E_sched._basis_for_layer(ℓ, d, device, dtype)
#                     if U is None or r is None or r < 1:
#                         deltas.append(torch.zeros_like(w)); continue
#                     keep = max(1, int(self.dropout_keep * r))
#                     idx  = torch.randperm(r, device=U.device)[:keep]
#                     dropped = torch.ones(r, device=U.device, dtype=torch.bool)
#                     dropped[idx] = False
#                     G = U.t() @ U + 1e-6 * torch.eye(r, device=U.device, dtype=U.dtype)
#                     alpha = torch.linalg.solve(G, U.t() @ w)                   # coeffs
#                     alpha_drop = alpha * dropped.float()
#                     deltas.append(-(U @ alpha_drop) * 0.2)                     # gentle removal
#                 # apply for this batch only
#                 with self.E_sched.apply_delta_temporarily(layer_param_groups, deltas):
#                     pass  # the rest continues with this temporary mask

#             # 5) optimizer step: either SAM-in-E or normal
#             if self.use_E_SAM and self.E_sched is not None and self.E_sched.active():
#                 # First backward to get grads
#                 self.opt.zero_grad()
#                 base_loss.backward()

#                 # Build grad flats per layer
#                 grad_flats = []
#                 for plist in layer_param_groups:
#                     g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).view(-1) for p in plist])
#                     grad_flats.append(g)

#                 # Project onto E and build ascent delta
#                 deltas = []
#                 L = len(grad_flats)
#                 for ℓ, g in enumerate(grad_flats):
#                     gE = self.E_sched.project_grad_to_E(g, ℓ, L)
#                     if gE.norm() > 0:
#                         delta = self.sam_eps * gE / (gE.norm() + 1e-12)
#                     else:
#                         delta = torch.zeros_like(g)
#                     deltas.append(delta)

#                 # Second forward/backward at adversarial point (temporary)
#                 with self.E_sched.apply_delta_temporarily(layer_param_groups, deltas):
#                     self.opt.zero_grad()
#                     pred_adv = self.model(image.float())
#                     loss_adv = loss_metric(pred_adv, label)
#                     loss_adv.backward()
#                 self.opt.step()
#                 loss_value = loss_adv.item()
#             else:
#                 self.opt.zero_grad()
#                 base_loss.backward()
#                 self.opt.step()
#                 loss_value = base_loss.item()
                
#             y_preds.extend(prediction.detach().cpu().view(-1).tolist())
#             y_trues.extend(label.detach().cpu().view(-1).tolist())

#             # print('prediction :', y_preds)
#             # print("accuracy is : {0:.16f}".format( metrics.accuracy_score(y_trues, y_preds)))
#             mae = mae_metric(prediction, 
#                             label)

#             # loss_value = base_loss.item()
#             losses.append(loss_value)
#             maes.append(mae.item())
#         # logging.getLogger(f"client.{self.client_id}").debug(
#         #     f"[train] epoch={epoch} loss={np.mean(losses):.4g} cons_pen={np.mean(cons_pens):.4g}")
#         # log_print(f"Epoch {epoch}: Loss={losses}, acc={aucs}", context="CLIENT TRAINING")

#         # --------- B) log/stash epoch-level extras so wrapper can return them ---------
#         mean_cons = float(np.mean(cons_pens)) if cons_pens else 0.0
#         logging.getLogger(f"client.{self.client_id}").debug(
#             f"[train] epoch={epoch} loss={np.mean(losses):.4g} mae={np.mean(maes):.4g} "
#             f"cons_pen={mean_cons:.4g} gradE_mean={gradE_mean}"
#         )
#         # expose for wrapper -> FitRes.metrics
#         self._last_epoch_consistency = mean_cons
#         self._last_epoch_gradE_mean  = gradE_mean
#         self._last_epoch_gradE_p95   = gradE_p95
#         # ------------------------------------------------------------------------------

#         return np.mean(losses), np.mean(maes)
        
    
#     @debug_function(context="CLIENT EVALUATION")
#     @torch.no_grad()
#     def evaluate_normal_model(self):
#         self.model.eval()

#         y_preds = []
#         y_trues = []
#         losses = []
#         maes = []
#         site_sums = defaultdict(float)
#         site_counts = defaultdict(int)
#         loss_metric = nn.MSELoss()
#         maes_metric = nn.L1Loss()

#         for i, (image, label, metadata) in enumerate(self.local_dataloader):
#             # if torch.cuda.is_available():
#             #     image = image.cuda()
#             #     label = label.cuda()
#             image = image.to(self.device_pref)
#             label = label.to(self.device_pref)
#             prediction = self.model(image.float())
#             # label = torch.squeeze(label, dim=[1])

#             loss = loss_metric(prediction, label)
            
#             err = (prediction - label).pow(2).view(-1).cpu()      # <-- NOT nn.MSELoss
#             sites = metadata[:, 0].long()                         # shape [B]
#             for e, s in  zip(err, sites):
#                 site_sums[s.item()] += e.item()
#                 site_counts[s.item()] += 1
            
#             y_pred = prediction.detach().cpu().numpy().flatten()
#             y_true = label.detach().cpu().numpy().flatten()

#             y_preds.extend(y_pred)
#             y_trues.extend(y_true)

#             mae = maes_metric(prediction, 
#                             label)

#             losses.append(loss.item())
#             maes.append(mae.item())

#         avg_loss = np.mean(losses)
#         avg_mae = np.mean(maes)
#         site_mse = {s: site_sums[s]/site_counts[s] for s in site_sums}
#         macro_mse = sum(site_mse.values())/len(site_mse)   # un-weighted mean
#         var_site, gap9010 = site_spread_stats(site_mse)
#         # logging.getLogger("validation").debug(f"[site_spread] var={var_site:.4f} gap90-10={gap9010:.4f}")

#         return avg_loss, avg_mae, site_mse, macro_mse, var_site, gap9010
# '''
#     #Disable DANN3D model training for now to reduce complexity.
    
    @debug_function(context="CLIENT")
    def train_dann3d_model(self, epoch, config=None): #(self, epoch, writer, log_every=100)

        _ = self.model.train()
        disc_old = [p.detach().clone() for p in self.model.domain_classifier.parameters()]
        y_preds = []
        y_trues = []
        d_losses = []
        p_losses = []
        losses = []
        maes = []
        ents = []
        ent_accs = []
        
        
        warmup_rounds = 3
        Bmin = self.standard_dt_size                 # smallest site size
        Bmax = self.max_dt_size                   # largest site size
        Bcur = self.data_size
        ramp_rounds   = 15
        lam_min, lam_max = 0.01, 0.10     # GRL coeff will go from 0.01 → 0.10
        # Merge per-round config with static cfg (YAML)
        _cfg = {**self.cfg, **(config)}
        smooth_eps = _cfg.get("smooth_eps", 0.01)
        total_rounds = config["num_rounds"]
        current_round = config["current_round"]
        current_total_epoch = (current_round-1)* self.epochs + epoch
        # p_epoch = current_total_epoch / (total_rounds * self.epochs)  # progress in [0,1] for the current round
        p = current_total_epoch / (total_rounds * self.epochs)  # progress in [0,1] for the current round        
        # if current_total_epoch < 10:             # Phase-A: task focus
        #     self.model.grl.coeff = 0.0
        # else:
            # lambda_d = 0.3 * (2/(1+math.exp(-10*p)) - 1) + 0.1
        coeff = _cfg.get("grl_coeff", 7.0) # * (2/(1+math.exp(-7*p_epoch)) - 1)
        # Phase A: pure prediction for the first warmup_rounds
        if current_round < warmup_rounds:
            self.model.grl.coeff = 0.0
        else:
            # Phase B: ramp GRL very gently
            self.model.grl.coeff = coeff * (2/(1+math.exp(-5*p)) - 1)
        MU_glob =  _cfg.get("mu_global", 20) #best: 20    # small global‐disc tether
        loss_metric = nn.MSELoss()
        auc_metric = nn.L1Loss()
        eps = smooth_eps
        domain_metric = nn.CrossEntropyLoss(label_smoothing=eps)  # label smoothing for domain classifier
        # if Bcur/ Bmax == 1.0:
        #     self.model.grl.coeff = 2.0
        #     MU_glob = 50.0
        print(f"Current total epoch: {current_total_epoch}, current round: {current_round}, total rounds: {total_rounds}")

        lambda_d = 0.0

        # # ------- one-time schedule after round 5 ------------------------------
        # if current_round > 5 and epoch == 0:           # only first epoch of the round
        #     # ❶ shrink GRL
        #     self.model.grl.coeff *= 0.50

        #     # ❷ shrink discriminator LR
        #     for pg in self.opt_disc.param_groups:
        #         pg['lr'] *= 0.50

        #     logging.getLogger(f"client.{self.client_id}").debug(
        #         f"[sched] round>{5}: coeff->{self.model.grl.coeff:.4g}, "
        #         f"d_lr->{self.opt_disc.param_groups[0]['lr']:.4g}"
        #     )
        # # ----------------------------------------------------------------------

        logging.getLogger(f"client.{self.client_id}").log(getattr(logging, "DEBUG", logging.DEBUG), f"CLIENT {self.client_id} GRL Coeff is: {self.model.grl.coeff}\n")
        logging.getLogger(f"client.{self.client_id}").log(getattr(logging, "DEBUG", logging.DEBUG), f"CLIENT {self.client_id} Label Smoothing is: {eps}\n")
        logging.getLogger(f"client.{self.client_id}").log(getattr(logging, "DEBUG", logging.DEBUG), f"CLIENT {self.client_id} MU global is: {MU_glob}\n")
        logging.getLogger(f"client.{self.client_id}").log(getattr(logging, "DEBUG", logging.DEBUG), f"CLIENT {self.client_id} d_lr is: {self.d_lr}\n")
        logging.getLogger(f"client.{self.client_id}").log(getattr(logging, "DEBUG", logging.DEBUG), f"CLIENT {self.client_id} lr is: {self.lr}\n")

        for i, (image, label, metadata) in enumerate(self.local_dataloader):
            
            domain_real = metadata[:, 0].long().to(image.device)   # shape (B,)

            # ---------- put this at the top of train_dann3d_model -------------
            import hashlib, numpy as np, torch

            def sha(flat):
                v = np.concatenate([p.flatten() for p in flat])
                return hashlib.sha1(v).hexdigest()

            if epoch == 0 and i == 0:        # first batch of the round
                flat_before = flatten_layer_param_list_for_flower(
                    self.model.get_params_for_layers()
                )
            if epoch == 4 and i == 0:        # first batch of the round
                flat_before = flatten_layer_param_list_for_flower(
                    self.model.get_params_for_layers()
                )
            # -------------------------------------------------------------------       
                 
            # ---- build the domain-label tensor ----
            # domain_label = torch.full(                    # shape (B,)
            #     (image.size(0),),                      # batch size
            #     self.client_id,                        # constant value
            #     dtype=torch.long
            # )  
            domain_label = domain_real
            # log_print(f"Epoch {epoch}, Batch {i}: image shape={image.shape}, label shape={label.shape}", context="CLIENT TRAINING")
            if torch.cuda.is_available():
                
                image = image.cuda()
                label = label.cuda()
                domain_label = domain_label.cuda()
                # weight = weight.cuda()

            prediction, domain_prediction = self.model(image.float())
            # with torch.no_grad():
            #     print("domain_pred.shape:", domain_prediction.shape)  # [B, C]
            #     print("domain_label.min/max:", int(domain_label.min()), int(domain_label.max()))
            #     assert domain_prediction.shape[1] > int(domain_label.max()), \
            #         f"label {int(domain_label.max())} >= C {domain_prediction.shape[1]}"
            #     exit()
            # prediction = prediction.view(-1)  # Flattens [batch_size, 1] → [batch_size]
            # label = label.view(-1)
            # print(prediction['y_pred'])
            # lable_box = Box({'y_trues': label})
            # print("prediction: ", prediction['y_pred'].shape)
            # print("lable: ", lable_box['y_trues'])
            prediction_loss = loss_metric(prediction, label)
            domain_loss = domain_metric(domain_prediction, domain_label.long())
            
            with torch.no_grad():
                sm = domain_prediction.softmax(1)
                ent = -(sm * sm.log()).sum(1).mean()
                acc = (sm.argmax(1) == domain_label).float().mean()
                # print('softmax entropy:',
                # -(sm * sm.log()).sum(1).mean().item(),
                # f"acc={acc*100:.1f}")   # in nats
                ents.append(ent.item())
                ent_accs.append(acc.item())

            # -------------------------------------------------------------
            #   Choose *either* FedProx (default) or SCAFFOLD
            # -------------------------------------------------------------
            prox = 0.0
            if self.scaffold_active:
                # ----- SCAFFOLD grad correction -----
                for p, ci, cg in zip(
                        self.model.domain_classifier.parameters(),
                        self.ci_disc,
                        self.c_global_disc,
                ):
                    if p.grad is not None:
                        p.grad.data = p.grad.data + torch.from_numpy(cg - ci).to(p.grad.device)
            else:
                # ---------- FedProx -----------------
                # MU_loc = 0 # 35e-3 * (Bcur / Bmax)**0.5

                # prox_loc  = sum((p - p0).pow(2).sum()
                #                 for p,p0 in zip(self.model.domain_classifier.parameters(), disc_old))
                # <-- new bit: tether to server’s global disc weights
                if self.ProxAll:
                    prox_feat = self.prox_term(self.model.featurizer.parameters(), self.feat_global, MU_glob)
                    prox_reg  = self.prox_term(self.model.regressor.parameters(),  self.reg_global,  MU_glob)
                    prox_disc = self.prox_term(self.model.domain_classifier.parameters(), disc_old, MU_glob)
                    prox = prox_feat + prox_reg + prox_disc
                else:
                    prox_glob = sum((p - pg).pow(2).sum()
                                    for p,pg in zip(self.model.domain_classifier.parameters(),
                                                self.disc_global))

                    prox = (MU_glob/2)*prox_glob # + (MU_loc/2)*prox_loc
            # log_print(f"Epoch {epoch}, Batch {i}: label is: {label}, prediction is: {prediction}", context="CLIENT TRAINING")
            # log_print(f"Epoch {epoch}, Batch {i}: Loss={loss.item():.4f}", context="CLIENT TRAINING")
            
            # print('prediction: ', prediction['y_pred'], "actual: ", lable_box['y_trues'])
            # loss = prediction_loss + lambda_d*domain_loss
            loss = prediction_loss + domain_loss + prox
            
            self.opt.zero_grad()
            if self.model.grl.coeff > 0:
                self.opt_disc.zero_grad()
            # print('loss is :', float(loss))
            # print(prediction['y_pred'].shape, lable_box['y_trues'].shape)
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(self.model.domain_classifier.parameters(), 2.0 * math.sqrt(Bcur / Bmax))
            torch.nn.utils.clip_grad_norm_(self.model.domain_classifier.parameters(), 5.0)

            

            self.opt.step()
            if self.model.grl.coeff > 0:
                self.opt_disc.step()

            
            y_preds.extend(prediction.detach().cpu().view(-1).tolist())
            y_trues.extend(label.detach().cpu().view(-1).tolist())

            # print('prediction :', y_preds)
            # print("accuracy is : {0:.16f}".format( metrics.accuracy_score(y_trues, y_preds)))
            acc = auc_metric(prediction, 
                            label)

            loss_value = loss.item()
            d_losses.append(domain_loss.item())
            p_losses.append(prediction_loss.item())
            losses.append(loss_value)
            maes.append(acc.item())

        # --------------- update ci for SCAFFOLD -----------
        if self.scaffold_active:
            eta   = self.lr
            K     = len(self.local_dataloader)
            theta_new = [p.detach().cpu().numpy() for p in self.model.domain_classifier.parameters()]
            theta_old = [p for p in disc_old]  # numpy already
            self.ci_disc = [
                ci + (old - new) / (eta * K)
                for ci, old, new in zip(self.ci_disc, theta_old, theta_new)
            ]


        return np.mean(losses), np.mean(maes), np.mean(d_losses), np.mean(p_losses), np.mean(ents), np.mean(ent_accs)
    
    @debug_function(context="CLIENT EVALUATION")
    @torch.no_grad()
    def evaluate_dann3d_model(self, config=None, scheduler=None):
        self.model.eval()

        y_preds = []
        y_trues = []
        # d_losses = []
        # p_losses = []
        losses = []
        maes = []
        site_sums = defaultdict(float)
        site_counts = defaultdict(int)
        total_rounds = config["num_rounds"]
        current_round = config["current_round"]
        loss_metric = nn.MSELoss()
        auc_metric = nn.L1Loss()
        # domain_metric = nn.CrossEntropyLoss()
        # progress = current_round / total_rounds

        for i, (image, label, metadata) in enumerate(self.local_dataloader):
            # domain_label = torch.full(                    # shape (B,)
            #     (image.size(0),),                      # batch size
            #     self.client_id,                        # constant value
            #     dtype=torch.long
            # )  
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()
                # domain_label = domain_label.cuda()

            prediction, _ = self.model(image.float())
            # label = torch.squeeze(label, dim=[1])

            prediction_loss = loss_metric(prediction, label)
            # domain_loss = domain_metric(domain_prediction, domain_label.long())
            
            err = (prediction - label).pow(2).view(-1).cpu()      # <-- NOT nn.MSELoss
            sites = metadata[:, 0].long()                         # shape [B]
            # domain_loss = domain_metric(domain_prediction, domain_label.long())
            for e, s in  zip(err, sites):
                site_sums[s.item()] += e.item()
                site_counts[s.item()] += 1
            
            
            y_pred = prediction.detach().cpu().numpy().flatten()
            y_true = label.detach().cpu().numpy().flatten()

            loss = prediction_loss

            y_preds.extend(y_pred)
            y_trues.extend(y_true)

            mae = auc_metric(prediction, 
                            label)
            
            # d_losses.append(domain_loss.item())
            # p_losses.append(prediction_loss.item())
            losses.append(loss.item())
            maes.append(mae.item())

        # avg_d_loss = np.mean(d_losses)
        # avg_p_loss = np.mean(p_losses)
        avg_loss = np.mean(losses)
        avg_mae = np.mean(maes)
        if scheduler is not None:
            scheduler.step(avg_loss)
        site_mse = {s: site_sums[s]/site_counts[s] for s in site_sums}
        macro_mse = sum(site_mse.values())/len(site_mse)   # un-weighted mean

        return avg_loss, avg_mae, site_mse, macro_mse



# Flower wrapper
import flwr as fl
class FlowerClientWrapper(fl.client.NumPyClient):
    def __init__(self, fed_client: FedClient, log_path: str):
        # log_print(f"[DEBUG] Initialized client {fed_client.client_id}")

        self.fed_client = fed_client
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)
        with open(os.path.join(self.log_path, f"client_{self.fed_client.client_id}_metrics.txt"), "a") as f:
            f.write(f"The Dataset Size is : {self.fed_client.data_size}\n")
            f.write(f"The Client ID is : {self.fed_client.client_id}\n")
        self.logger = self.setup_per_client_logging(self.fed_client.client_id)
        self.logger.info("Client %s started; dataset = %d", fed_client.client_id, fed_client.data_size)
        
    def setup_per_client_logging(self, client_id: int) -> logging.Logger:
        """
        Create (or reuse) a logger that writes *only* this client's records
        to logs/client_{id}.log.  Keeps your root logger untouched.
        """
        log_dir = self.log_path
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(log_dir, f"client_{client_id}.log")

        # 1️⃣ create / fetch logger with a unique name
        clog = logging.getLogger(f"client.{client_id}")
        clog.setLevel(logging.DEBUG)
        clog.propagate = False              # <- detach from root

        # 2️⃣ remove any handlers left from a previous init in this actor
        clog.handlers.clear()

        # 3️⃣ file handler
        fh = logging.FileHandler(log_path, mode="a")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        clog.addHandler(fh)

        # 4️⃣ optionally pipe Flower's debug into the same file
        flower = logging.getLogger("flwr")          # or "flwr.common"
        flower.setLevel(logging.DEBUG)
        flower.propagate = False
        if all(h.baseFilename != fh.baseFilename for h in flower.handlers
            if isinstance(h, logging.FileHandler)):
            flower.addHandler(fh)

        return clog
    # def attach_flower_filehandler(self, client_id):
    #     path = os.path.join(self.log_path, f"flower_client_{client_id}.log")
    #     fh = logging.FileHandler(path, mode="a")
    #     fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    #     logging.getLogger("flwr").addHandler(fh)
        
    @debug_function(context="CLIENT WRAPPER")
    def get_parameters(self, config=None):
        theta_flat = flatten_layer_param_list_for_flower(
            self.fed_client.get_params()
        )
        if self.fed_client.scaffold_active:
            ci_flat = flatten_layer_param_list_for_flower(self.fed_client.ci_disc)
            return theta_flat + ci_flat
        else:
            return theta_flat
    
    @debug_function(context="CLIENT WRAPPER")
    def set_parameters(self, parameters):
        # log_print("the parameters type is : ", type(parameters), "the parameter shape is : ", len(parameters) , context="CLIENT")
        # torch_params = flatten_layer_param_list(parameters)
        # self.fed_client.receive_and_update_params(parameters)
        n_ref = self.fed_client.n_ref_layers
        # log_print(f"FedClient.n_ref_layers: {n_ref}", context="CLIENT WRAPPER")
        # log_print(f"Received parameters length: {len(parameters)}", context="CLIENT WRAPPER")
        if n_ref is None:
            raise RuntimeError("FedClient.n_ref_layers not initialised")
 
        if len(parameters) == n_ref:
            # -------- vanilla / FedProx -------------
            self.fed_client.scaffold_active = False
            theta_flat = parameters
            self.fed_client.receive_and_update_params(theta_flat)
            
        elif len(parameters) == 2 * n_ref:
            # ---------- SCAFFOLD ----------
            self.fed_client.scaffold_active = True
            theta_flat = parameters[:n_ref]
            c_flat     = parameters[n_ref:]
            self.fed_client.receive_and_update_params(theta_flat)
            self.fed_client.c_global_disc = flatten_layer_param_list_for_model(c_flat)
            # initialise local ci once
            if self.fed_client.ci_disc is None:
                self.fed_client.ci_disc = [
                    np.zeros_like(x) for x in self.fed_client.c_global_disc
                ]
        else:
            raise ValueError(
                f"Parameter length {len(parameters)} does not match "
                f"expected {n_ref} or {2*n_ref}"
            )

    @debug_function(context="CLIENT WRAPPER")
    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        logs = self.fed_client.local_train_step(config=config)
        current_round = config.get("current_round", 0)
        # logging
        writer = SummaryWriter(log_dir=self.log_path)
        with open(os.path.join(self.log_path, f"client_{self.fed_client.client_id}_metrics.txt"), "a") as f:
            f.write(f"Training Round {current_round}\n")
            if(len (logs[0]) <= 3):
                for epoch, loss, mae in logs:
                    f.write(f"Epoch {epoch + 1}: Loss={loss:.4f}, MAE={mae:.4f}\n")
                    writer.add_scalar("Train/Loss", loss, epoch)
                    writer.add_scalar("Train/MAE", mae, epoch)
                f.write(f"Training Round Done\n")
            elif(len(logs[0])>3):
                for epoch, loss, mae, d_loss, p_loss, ent, ent_acc in logs:
                    f.write(f"Epoch {epoch + 1}: Loss={loss:.4f}, MAE={mae:.4f}, Domain Loss={d_loss:.4f}, Prediction Loss={p_loss:.4f}, Entropy={ent:.4f}, Entropy Accuracy={ent_acc:.4f}\n")
                    writer.add_scalar("Train/Loss", loss, epoch)
                    writer.add_scalar("Train/MAE", mae, epoch)
                    writer.add_scalar("Train/Domain Loss", d_loss, epoch)
                    writer.add_scalar("Train/Prediction Loss", p_loss, epoch)
                f.write(f"Training Round Done\n")
        writer.close()
        
        metrics = {
            "_id": self.fed_client.client_id,
            "loss": logs[-1][1] if len(logs[-1]) > 1 else logs[-1][0],
            "mae":  logs[-1][2] if len(logs[-1]) > 2 else logs[-1][1],
            "num_samples": len(self.fed_client.local_data),
        }
        # add optional diagnostics
        for k in ("_last_epoch_consistency", "_last_epoch_gradE_mean", "_last_epoch_gradE_p95"):
            val = getattr(self.fed_client, k, None)
            if val is not None:
                metrics[k] = float(val)

        # log_print("end of fit for client : ", self.fed_client.client_id, context="CLIENT WRAPPER")
        return self.get_parameters(), len(self.fed_client.local_data), metrics

    @debug_function(context="CLIENT WRAPPER")
    def evaluate(self, parameters, config=None):
        return 0.0, len(self.fed_client.local_data), {"accuracy": 0.0}