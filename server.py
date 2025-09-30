from multiprocessing import context
import torch
import torch.optim as optim
import os
import time
from model import BrainCancer
from model_feature_regress import DANN3D, BrainCancerFeaturizer, BrainCancerRegressor
from utils import get_layer_params_list, get_layer_params_dict, flatten_layer_param_list_for_model, reconstruct_layer_from_flat
from utils import debug_function, log_print
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import math
import pandas as pd
import torch.nn.functional as F
import base64, json, zlib
# at the top of files that use them
from utils import (
    gradE_ratio_for_layers,
    robust_mae_under_E,
    robustness_curve_auc,
    directional_sensitivity_along_basis,
    basis_explained_energy,
    site_spread_stats,
)


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

class Server:
    @debug_function(context="SERVER")
    def __init__(self, num_clients, val_dataset, test_dataset, model_type="Normal", alpha_var=0.1, beta_sparsity=0.01, run_id=None, num_rounds=10):
        """
        num_clients: number of clients in the federation
        alpha_var: regularization weight for variance minimization in the M_inv block
        beta_sparsity: regularization weight for L1 (sparsity) in the M_spec block
        default: alpha_var=0.1, beta_sparsity=0.1
        new 1: alpha_var=0.1, beta_sparsity=0.01
        """
        # hook = sy.TorchHook(torch)  # Hook PyTorch
        self.run_id = run_id
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_clients = num_clients
        self.alpha_var = alpha_var
        self.beta_sparsity = beta_sparsity
        self.num_rounds = num_rounds
        self.best_score = -np.inf
        self.best_layer = None
        self.domains = []
        num_cpus = get_num_cpus()
        torch.set_num_threads(num_cpus)

        if model_type == "Normal":
            self.initial_dummy_model = BrainCancer()
            self.initial_dummy_paramters_dict = get_layer_params_dict(self.initial_dummy_model)
            self.initial_dummy_paramters_list = get_layer_params_list(self.initial_dummy_model)
        elif model_type == "DANN3D":
            # n_domains  = 15
            n_domains = 62
            feat_net   = BrainCancerFeaturizer(use_conv5=True)  # or False for conv4
            reg_head   = BrainCancerRegressor()
            self.initial_dummy_model = DANN3D(feat_net, reg_head, n_domains, hidden_size=512)
            self.initial_dummy_paramters_dict = get_layer_params_dict(self.initial_dummy_model)
            self.initial_dummy_paramters_list = get_layer_params_list(self.initial_dummy_model)
        self.num_layers = len(self.initial_dummy_paramters_list)
        self.layer_slices = []            # one slice per layer
        start = 0
        for layer_param_list in self.initial_dummy_paramters_list:
            layer_size = sum(p.numel() for p in layer_param_list)
            self.layer_slices.append(slice(start, start + layer_size))
            start += layer_size
            
        # one-time debug to verify the last layer really is the output conv
        for i, tensors in enumerate(self.initial_dummy_paramters_list):
            shapes = [tuple(t.shape) for t in tensors]
            log_print(f"Layer {i}: {shapes}", context="OUTPUT LAYER CHECK")
            
            
        # Example: layer_slices[4] points to the regressor-FC weights
        self.E_layer_wise = {}
        self.shared_mean_layers = {}     # {layer_idx: Tensor[d]}
        self.residual_bank_mode = "accumulate" # or "accumulate"
        self.max_bank_rounds = 3         # only used if accumulate (caps memory)
        self.prev_spec_global = {}  # {layer_idx: Tensor [d_l]}  for each layer
        self.inv_agg = {}
        self.SERVER_LOG_HEADERS = [
            "round",
            "layer_idx",
            "inv_variance",
            "spec_l1_norm",
            "recon_loss",
            "inv_norm",
            "spec_norm",
            "agg_norm",
            "param_diversity",
            "probe_accuracy_inv",
            "probe_accuracy_spec",
            "cos_to_mean",
            "spread",
            "angle_inv",
            "pc1_coords"

        ]

        # self.vms = []
        self.client_log_file_paths = []
        # Generate a unique run directory (create if doesn't exist)
        if self.run_id is None:
            self.run_id = time.strftime("%Y-%m-%d_%H-%M-%S")  # Timestamp-based run ID
        self.run_dir = os.path.join("./runs", f"run_{run_id}")  # Each run has a separate directory
        os.makedirs(self.run_dir, exist_ok=True)

        self.server_log_dir = os.path.join(self.run_dir, "server_log")
        os.makedirs(self.server_log_dir, exist_ok=True)  # Create directory if it doesn't exist
        self.server_log_path = os.path.join(self.server_log_dir, "server_metrics.csv")
        self.validation_log_path = os.path.join(self.server_log_dir, "validation_metrics.txt")
        self.sensitivity_log_path = os.path.join(self.server_log_dir, "sensitivity_metrics.csv")
        self.winner_log_path = os.path.join(self.server_log_dir, "winner_round.csv")
        self.text_log_path = os.path.join(self.server_log_dir, "test_metrics.txt")
        self.initialize_server_logger()
        # Create virtual machines for each client
        for i in range(num_clients):
            # self.vms[i] = sy.VirtualMachine(name="domain_{i}")
            # self.domains[i] = self.vms[i].get_root_client()
            # Define file path for logging
            self.client_log_dir = os.path.join(self.run_dir, "clients_log")
            os.makedirs(self.client_log_dir, exist_ok=True)  # Create directory if it doesn't exist
            self.client_log_file_paths.append(os.path.join(self.client_log_dir, f"client_{i}_metrics"))

        self.r_shared = 2       # rank of shared subspace per layer (1–2 for K=5)
        self.r_y = 1            # how many label-aligned directions to keep
        self.output_layer_idx = self.num_layers - 1  # or 4 if you know "regressor-FC is layer 4"
        # Optional: store aggregated domain-invariant params for reference
        # { layer_index: [ aggregated_invariant_vector ] }
        self.global_invariant_store = {}
       
    def initialize_server_logger(self):
        # os.makedirs(self.server_log_dir, exist_ok=True)
        if not os.path.exists(self.server_log_path):
            with open(self.server_log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.SERVER_LOG_HEADERS)
    


    def log_server_metrics(self, row_dict):
        with open(self.server_log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row_dict[h] for h in self.SERVER_LOG_HEADERS])

    def _pack_b64(self, obj: dict) -> str:
        """JSON -> zlib -> base64 (ascii string)."""
        raw = json.dumps(obj).encode("utf-8")
        return base64.b64encode(zlib.compress(raw)).decode("ascii")

    # residual annealing schedule
    def residual_gamma(self, round_idx: int) -> float:
        # warmup + decay: FedAvg for first 2 rounds, then decay to 0
        if round_idx <= 2:
            return 1.0
        if round_idx <= 6:
            return 0.75
        if round_idx <= 10:
            return 0.5
        if round_idx <= 14:
            return 0.25
        return 0.25

    def _weights_vec(self, all_client_ids, weights_by_cid, device, dtype):
        K = len(all_client_ids)
        if not weights_by_cid:
            return torch.full((K,), 1.0 / K, device=device, dtype=dtype)
        w = torch.tensor([weights_by_cid[c] for c in all_client_ids], device=device, dtype=dtype)
        s = w.sum().clamp_min(1e-12)
        return w / s


    def residual_matrix_for_layer(self, layer_idx: int, K_expected: int, fallback: torch.Tensor = None):
        """
        Return a d×K view of residuals for metrics.
        - Uses bank if available; otherwise uses `fallback` (typically the current round's E).
        - Ensures the returned matrix has exactly K_expected columns (slice/pad).
        """
        E = self.E_layer_wise.get(layer_idx, None)
        if E is None:
            if fallback is None:
                raise KeyError(f"E_layer_wise[{layer_idx}] not set and no fallback provided")
            E = fallback.detach()

        d, K_have = E.shape
        if K_have > K_expected:
            return E[:, -K_expected:]                      # most recent K columns
        if K_have < K_expected:
            # pad by repeating the last column (rare)
            if K_have == 0:
                pad = torch.zeros(d, K_expected, device=E.device, dtype=E.dtype)
                return pad
            pad = E[:, -1:].repeat(1, K_expected - K_have)
            return torch.cat([E, pad], dim=1)
        return E

    def compute_thresholds(self):
        srv = pd.read_csv(self.server_log_path)

        latest_round = srv['round'].max()
        # mean_angle_inv = srv['angle_inv'].mean()
        # mean_spread = srv['spread'].mean()
        window = srv[srv['round'] >= max(1, latest_round - 2)]

        # Slightly relax thresholds to avoid aggressive updates
        # dynamic thresholds
        threshold_angle  = window['angle_inv'].mean()  * 0.95
        threshold_spread = window['spread'].mean()     * 0.95

        return threshold_angle, threshold_spread


    @debug_function(context="SERVER")
    def adapt_alpha_beta(self):
        threshold_angle, threshold_spread = self.compute_thresholds()

        srv = pd.read_csv(self.server_log_path)
        # Focus on the latest round
        latest_round = srv['round'].max()
        window = srv[srv['round'] >= max(1, latest_round - 2)]
        # latest_data = srv[srv['round'] == latest_round]

        # Compute mean across layers for the latest round
        latest_data = window[window['round'] == latest_round]
        current_angle_inv = latest_data['angle_inv'].mean()
        current_spread = latest_data['spread'].mean()

        # # Adapt alpha_var
        # if current_angle_inv < threshold_angle:
        #     self.alpha_var *= 1.01  # increase invariance strength
        # else:
        #     self.alpha_var *= 0.99  # relax invariance slightly

        # warm-up: skip first 5 rounds
        if latest_round > 5:
            # only increase invariance
            if current_angle_inv < 0.8 * threshold_angle:
                self.alpha_var = min(self.alpha_var * 1.05, 0.5)

            # only decrease sparsity (i.e. allow more spread)
            if current_spread < 0.8 * threshold_spread:
                self.beta_sparsity = max(self.beta_sparsity * 0.95, 0.001)

        # # Keep parameters within reasonable bounds
        # self.alpha_var = min(max(self.alpha_var, 0.01), 1.0)
        # self.beta_sparsity = min(max(self.beta_sparsity, 0.001), 0.1)
        return 
        # # Optionally log the adaptive changes:
        # with open(self.server_log_dir + '/adaptive_params.csv', 'a') as f:
        #     f.write(f"{latest_round},{self.alpha_var:.4f},{self.beta_sparsity:.4f}\n")  
  
    @debug_function(context="SERVER")
    @torch.no_grad()
    def pc1_coords(self, M_spec):
        # 2) separate  …  you already have  M_spec,  M_inv
        M0 = M_spec - M_spec.mean(1, keepdim=True)      # (d, K)

        # --- rank-1 SVD ---------------------------------------------------
        # returns U (d×1), S (1,), V (K×1)
        U, S, V = torch.svd_lowrank(M0, q=1)            # q = 1 → top component
        pc1_scanner = V[:, 0]                           # shape (K,)
        # scores of each scanner on PC-1  (proportional to pc1_scanner)
        coords = (S[0] * pc1_scanner).tolist()          # same length K
        return coords

    @debug_function(context="SERVER")
    @torch.no_grad()
    def _build_E_basis_b64(self, r: int = 1, use_float16: bool = True) -> str:
        """
        Build per-layer basis and return ONE compact string suitable for FitIns.config.
        Each layer stores float16 bytes to keep it small.
        """
        payload = {}
        for ℓ, E in self.E_layer_wise.items():
            X = E - E.mean(dim=1, keepdim=True)           # [d, K]
            q = min(r, X.shape[1])
            if q < 1 or X.abs().sum() == 0:
                continue
            U, S, V = torch.svd_lowrank(X, q=q)          # U:[d,q]
            scale = (S[0] / (float(X.shape[1]) ** 0.5 + 1e-8)).item()

            U_np = U[:, :q].cpu().numpy()
            if use_float16:
                U_np = U_np.astype(np.float16)            # big shrink
                dtype = "float16"
            else:
                dtype = "float32"

            payload[str(ℓ)] = {
                "shape": list(U_np.shape),
                "dtype": dtype,
                "scale": float(scale),
                "u_b64": base64.b64encode(U_np.tobytes()).decode("ascii"),
            }

        # One scalar string for the entire dict
        return self._pack_b64(payload)

    @debug_function(context="SERVER")
    @torch.no_grad()
    def build_E_basis_dict(self, r: int = 1) -> dict:
        """
        Build a small per-layer basis from the residual bank E (shape [d,K]).
        Returns a JSON-serializable dict:
        {
            "0": {"U": [[...],[...],...], "scale": 0.37},
            "1": {"U": [[...],[...],...], "scale": 0.22},
            ...
        }
        Keys are strings (safer for JSON). U has shape [d, r_use] per layer.
        'scale' is a sensible default step-size for the scheduler (std along PC1).
        """
        payload = {}
        for ℓ, E in self.E_layer_wise.items():
            # Center across clients
            X = E - E.mean(dim=1, keepdim=True)        # [d, K]
            q = min(r, X.shape[1])
            if q < 1 or X.abs().sum() == 0:
                continue
            U, S, V = torch.svd_lowrank(X, q=q)        # U:[d,q], S:[q], V:[K,q]
            # A reasonable scale = std along the first component in client-space
            # (same as S[0]/sqrt(K)); guard small K
            scale = (S[0] / (float(X.shape[1]) ** 0.5 + 1e-8)).item()
            payload[str(ℓ)] = {
                "U": U[:, :q].cpu().numpy().tolist(),  # JSON-friendly
                "scale": float(scale),
            }
        return payload


    @torch.no_grad()
    def build_E_bank_payload(self, dtype=np.float32) -> dict:
        """
        Heavier payload: send the full residual bank per layer.
        {
        "0": [[...column0...], [...column1...], ...],   # actually [d,K] but list-of-lists
        "1": ...
        }
        Only use this if you’re OK with larger network traffic.
        """
        payload = {}
        for ℓ, E in self.E_layer_wise.items():
            payload[str(ℓ)] = E.detach().cpu().numpy().astype(dtype).tolist()
        return payload

    @debug_function(context="SERVER")
    def export_scheduler_config(self,
                                mode: str = "pc",
                                aug: str = "none",  # "consistency" | "sam" | "dropout" | "irm_fd"
                                rank: int = 1,
                                scale_override: float = None,
                                zero_last_layer: bool = True,
                                send_basis: bool = True,
                                send_bank: bool = False) -> dict:
        """
        Build the config dict that clients will read in local_train_step().
        Set `send_basis` or `send_bank` depending on what you want to transmit.
        """
        cfg = {
            "E_mode": mode,                 # "off" | "pc" | "dirichlet" | "lowrank"
            "E_client_aug": aug,         # "consistency" | "sam" | "dropout" | "irm_fd"
            "E_rank": int(rank),            # used by scheduler
            "E_zero_last_layer": bool(zero_last_layer),
        }
        if mode == "off":
            # No E_basis or E_bank sent; clients will not use residuals
            return cfg
        if send_basis:
            basis_b64 = self._build_E_basis_b64(r=rank, use_float16=True)
            cfg["E_basis"] = basis_b64
            if scale_override is not None:
                cfg["E_scale"] = float(scale_override)

        # if send_bank:
        #     cfg["E_bank"] = self.build_E_bank_payload()

        return cfg
     
    @debug_function(context="SERVER")  
    @torch.no_grad()
    def get_output_head_across_clients(self, client_params_dict):
        """
        Returns (W_out, b) stacked across clients:
        W_out: Tensor [D, K]  (last-layer weight per client as column)
        b    : Tensor [K]     (last-layer bias per client)
        Assumes regression last layer has weight shape [1, D] (or [D_out, D]) and bias [D_out].
        If multiple outputs, we flatten to D = out_dim * in_dim but still useful for co-variation.
        """
        K = len(client_params_dict)
        all_client_ids = list(client_params_dict.keys())
        W_cols = []
        b_vals = []

        for cid in all_client_ids:
            # Last layer tensors for this client (list[Tensor])
            last_layer_tensors = client_params_dict[cid][self.output_layer_idx]

            # Heuristic: find the 2D weight and 1D bias in this layer's tensor list
            weight = None
            bias = None
            for t in last_layer_tensors:
                if t.dim() >= 2 and weight is None:
                    weight = t.detach().reshape(-1)  # flatten to [D]
                elif t.dim() == 1 and bias is None:
                    bias = t.detach().reshape(-1)   # [out] (often length 1)

            # Fallback: if somehow we didn’t find a single weight tensor, try concatenating
            if weight is None:
                multi = [tt.detach().reshape(-1) for tt in layer_tensors if tt.dim() >= 2]
                if len(multi) > 0:
                    weight = torch.cat(multi, dim=0)
                else:
                    raise RuntimeError(
                        f"Could not find weight-like tensor (dim>=2) in output layer "
                        f"{self.output_layer_idx} for client {cid}"
                    )

            if bias is None:
                bias = torch.zeros(1, device=weight.device, dtype=weight.dtype)

            W_cols.append(weight)
            # For multi-output, you can average bias or keep the first; here we take mean
            b_vals.append(bias.mean())

        # Sanity: all columns must have same D
        D = W_cols[0].numel()
        for j, w in enumerate(W_cols):
            if w.numel() != D:
                raise RuntimeError(
                    f"Output weight length mismatch across clients: client {all_client_ids[j]} has {w.numel()}, "
                    f"expected {D}"
                )


        W_out = torch.stack(W_cols, dim=1)  # [D, K]
        b = torch.stack(b_vals)             # [K]
        return W_out, b
      
    @debug_function(context="SERVER")
    @torch.no_grad()
    def lowrank_shared_split(self, Theta, w, r_shared=2, energy_thresh=0.95):
        """
        Theta: [d, K], w: [K] normalized weights (sum=1)
        Returns L (shared) and E (idiosyncratic), with:
        - L centered under weights (sum_j w_j * L[:, j] = 0),
        - energy-based rank selection in weighted client space.
        """
        d, K = Theta.shape
        # weighted mean and centering
        mu = Theta @ w                     # [d]
        Xc = Theta - mu[:, None]           # [d, K]

        # whiten columns by sqrt(w) to use standard dot as weighted dot
        sw = w.clamp_min(1e-12).sqrt()     # [K]
        Y  = Xc * sw[None, :]              # [d, K]

        # Gram in client space
        G = (Y.T @ Y) / max(1, d)          # [K, K]
        evals, V = torch.linalg.eigh(G)    # ascending
        evals = torch.clamp(evals, min=1e-12)
        order = torch.argsort(evals, descending=True)
        evals = evals[order]; V = V[:, order]

        # energy-based rank (cap at K-1)
        max_r = max(1, min(K - 1, r_shared if r_shared is not None else K - 1))
        cum = torch.cumsum(evals, dim=0); total = cum[-1]
        r = int(torch.searchsorted(cum, energy_thresh * total).item()) + 1
        r = max(1, min(r, max_r))

        V_r = V[:, :r]                     # [K, r]
        P   = V_r @ V_r.T                  # projector in Y-space

        # project and unwhiten back to X-space
        Yp = Y @ P                         # [d, K]
        Lc = Yp / sw[None, :]              # [d, K] columnwise divide

        # enforce weighted zero-mean (numerical guard)
        Lc = Lc - (Lc @ w)[:, None]

        # add back weighted mean
        L = Lc + mu[:, None]               # [d, K]
        E = Theta - L
        return L, E

    @debug_function(context="SERVER")
    @torch.no_grad()
    def split_label_aligned(self, L, W_out, b, w, r_y=1):
        """
        L    : [d, K] shared block
        W_out: [D, K] flattened output-head weights
        b    : [K]    head bias
        w    : [K]    normalized weights
        """
        d, K = L.shape
        sw = w.clamp_min(1e-12).sqrt()

        # weighted centering
        Lc = L - (L @ w)[:, None]                    # [d, K]
        Wc = W_out - (W_out @ w)[:, None]            # [D, K]
        bc = b - (w * b).sum()                       # scalar → expand later

        T  = torch.cat([Wc, bc.unsqueeze(0)], dim=0) # [D+1, K]
        Tw = T * sw[None, :]                         # weighted features

        # small-K SVD in weighted client space
        U, S, Vh = torch.linalg.svd(Tw.t(), full_matrices=False)  # U:[K,K]
        r = min(max(1, r_y), U.shape[1], K)
        U_r = U[:, :r]                                           # [K, r]
        Pi  = U_r @ U_r.T                                        # [K, K]

        # project L with the same weighted metric
        Y   = Lc * sw[None, :]
        Yp  = Y @ Pi
        M_y_centered = Yp / sw[None, :]

        M_inv = L - M_y_centered
        M_y   = L - M_inv
        return M_y, M_inv

    


      
    @debug_function(context="SERVER")    
    def weight_space_probe_aug(self, M_block, n_aug=64, noise_std=0.02):
        """
        Create n_aug noisy copies of each column, then fit logistic regression.
        """
        d, K = M_block.shape
        feats, labels = [], []
        for j in range(K):
            base = M_block[:, j]
            for _ in range(n_aug):
                z = base + noise_std * torch.randn_like(base)
                feats.append(z.cpu().numpy())
                labels.append(j)
        X = np.vstack(feats)
        y = np.array(labels)
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
        clf.fit(X, y)
        return clf.score(X, y)          # training==testing OK (lots of samples)

    @debug_function(context="SERVER")
    def alpha_mix(self, M_spec: torch.Tensor, layer_idx: int, alpha: float = 0.8):
        """
        Blend each client's M_spec[:, j] with previous global spec.
        - M_spec shape: [d, K]
        - prev_spec_global[layer_idx]: shape [d]
        - Returns: blended_spec [d, K]
        """
        d, K = M_spec.shape
        if layer_idx not in self.prev_spec_global:
            # If this is the first round, fallback to current mean
            self.prev_spec_global[layer_idx] = M_spec.mean(dim=1).clone().detach()

        global_spec = self.prev_spec_global[layer_idx].unsqueeze(1)  # [d, 1]
        blended_spec = alpha * M_spec + (1 - alpha) * global_spec

        # Update for next round: new global spec = weighted avg of current
        self.prev_spec_global[layer_idx] = blended_spec.mean(dim=1).detach()

        return blended_spec
            
    @debug_function(context="SERVER")
    def gather_client_params(self, client_params_dict, layer_idx):
        """
        client_params_dict: { client_id: [layer0_tensor, layer1_tensor, ...] }
        layer_idx: which layer to gather

        Returns a matrix param_matrix of shape (d_layer, K), where d_layer is 
        the flattened dimension of this layer, K = num_clients.
        """
        # for client_id, layer_params in client_params_dict.items():
        #     log_print(f"[DEBUG] Client {client_id} → Layer {layer_idx} total layers: {len(layer_params)}")
        #     for i, layer in enumerate(layer_params):
        #         log_print(f"[DEBUG] Client {client_id} → Layer {i}  len: {len(layer)}, len item 1: {len(layer[0])}, len item 2: {len(layer[1])}")
        param_list = []
        for client_id, layer_params in client_params_dict.items():
            flattened_tensor = torch.cat([
                p.view(-1) for p in layer_params[layer_idx]  # flatten each tensor in layer
            ])            # layer_tensor = param_layer_flatten[layer_idx]      # flatten
            # log_print(f"[DEBUG] Client {client_id} → Layer {layer_idx} param shape after flatten: {len(flattened_tensor)}", context="GATHER CLIENT PARAMS")
            param_list.append(flattened_tensor)
        
        if not param_list:
            raise ValueError(f"[ERROR] Empty param_list for layer {layer_idx}. client_params_dict keys: {list(client_params_dict.keys())}")
        # Stack all columns => shape (d_layer, K)
        # for i, p in enumerate(param_list):
        #     log_print(f"[DEBUG] {i}th param_list shape: {len(p)}")
        #     log_print(f"param[{i}] flattened size = {[t.numel() for t in p]}")
        #     log_print(f"total flattened vector size = {sum(t.numel() for t in p)}") 
        param_matrix = torch.stack(param_list, dim=1)
        return param_matrix

    @debug_function(context="SERVER")
    @torch.no_grad()
    def separate_weightspace(self, param_matrix, W_out, b, w, layer_idx):
        """
        param_matrix: [d, K] for layer `layer_idx`.
        W_out, b    : from get_output_head_across_clients()

        Returns:
        M_inv [d, K], M_y [d, K], E [d, K], metrics (dict)
        """
        # 1) Low-rank shared vs idiosyncratic
        L, E = self.lowrank_shared_split(param_matrix, w, r_shared=self.r_shared)

        # 2) Within shared, extract label-aligned vs invariant
        M_y, M_inv = self.split_label_aligned(L, W_out, b, w, r_y=self.r_y)

        # 3) Some diagnostics roughly analogous to your logs
        metrics = {}
        metrics["recon_loss"] = float(((M_inv + M_y + E) - param_matrix).pow(2).mean().item())
        metrics["inv_norm"]   = float(M_inv.norm().item())
        metrics["label_norm"] = float(M_y.norm().item())
        metrics["idios_norm"] = float(E.norm().item())
        # probe accuracies (how linearly separable columns are)
        K_now = param_matrix.shape[1]
        M_spec_like =  self.residual_matrix_for_layer(layer_idx, K_now, fallback=E) # residual plays the role of your old M_spec
        try:
            metrics["probe_accuracy_inv"]  = float(self.weight_space_probe_aug(M_inv))
            metrics["probe_accuracy_spec"] = float(self.weight_space_probe_aug(M_spec_like))
        except Exception:
            metrics["probe_accuracy_inv"]  = 0.0
            metrics["probe_accuracy_spec"] = 0.0

        # spread/angle/cos (use your previous definitions where sensible)
        with torch.no_grad():
            K = param_matrix.shape[1]
            M_spec_like = E  # residual plays the role of your old M_spec
            coords = self.pc1_coords(M_spec_like)
            # cosine to mean
            cos_to_mean = F.cosine_similarity(M_spec_like.mean(1), M_spec_like[:, 0], dim=0)
            spread = (M_spec_like - M_spec_like.mean(1, keepdim=True)).norm(dim=0).mean()
            angle_inv = torch.acos(torch.clamp(
                (M_inv * M_spec_like).sum() / (M_inv.norm() * M_spec_like.norm() + 1e-8),
                -1 + 1e-6, 1 - 1e-6
            ))
        metrics["cos_to_mean"] = float(cos_to_mean.item())
        metrics["spread"]      = float(spread.item())
        metrics["angle_inv"]   = float(angle_inv.item())
        metrics["pc1_coords"]  = coords

        return M_inv, M_y, E, metrics


    @debug_function(context="SERVER")
    def server_round(self, client_params_dict, num_layers, server_round, client_weights=None):
        """
        client_params_dict: { cid: [layer0_tensors(list), layer1_tensors(list), ...] }
        Returns updated_client_params: same structure as input, reshaped by reconstruct_layer_from_flat()
        """
        # for cid in client_params_dict:
        #     log_print(f"[SERVER ROUND] Client {cid} params: {len(client_params_dict[cid])}")
        updated_client_params = {cid: [] for cid in client_params_dict}
        all_client_ids = list(client_params_dict.keys())

        # 0) Grab output head across clients ONCE (used for every layer's label-alignment)
        W_out, b = self.get_output_head_across_clients(client_params_dict)  # [D,K], [K]



        for layer_idx in range(num_layers):
            # 1) gather
            param_matrix = self.gather_client_params(client_params_dict, layer_idx)
            device, dtype = param_matrix.device, param_matrix.dtype
            w = self._weights_vec(all_client_ids, client_weights, device, dtype)  # <- NEW

            # ---- Pure FedAvg warm-up (weighted) for first 2 rounds ----
            if server_round < 2:
                inv_agg = param_matrix @ w                      # [d], weighted mean
                E_blend = param_matrix - inv_agg[:, None]       # residual wrt weighted mean
                self.shared_mean_layers[layer_idx] = inv_agg.detach().clone()
                gamma = self.residual_gamma(server_round)
                for j, cid in enumerate(all_client_ids):
                    new_layer_flat = inv_agg + gamma * E_blend[:, j]
                    ref = client_params_dict[cid][layer_idx]
                    updated_client_params[cid].append(reconstruct_layer_from_flat(new_layer_flat, ref))
                continue

            # 2) decompose weight-space
            M_inv, M_y, E, metric_dict = self.separate_weightspace(param_matrix, W_out, b, w, layer_idx)

            # 3) aggregate shared (mean across clients)
            shared = M_inv + M_y                                         # [d, K]
            inv_agg = shared @ w                                # [d]

            # Sanity check: should be close to fedAvg mean
            if server_round <= 3:
                fedavg_flat = param_matrix @ w
                delta = (fedavg_flat - inv_agg).abs().max().item()
                log_print(f"[CHECK][round {server_round}][layer {layer_idx}] max|FedAvg_w - shared_mean| = {delta:.3e}", context="SANITY CHECK")

            # --- store shared mean for clients / unseen-domain use
            self.shared_mean_layers[layer_idx] = inv_agg.detach().clone()

            # --- choose what to store as residual bank: raw E (recommended) or EMA-blended
            E_to_store = E.detach().clone()               # or: self.alpha_mix(E, layer_idx, alpha=0.8).detach()

            if self.residual_bank_mode == "accumulate":
                prev = self.E_layer_wise.get(layer_idx)
                if prev is None:
                    self.E_layer_wise[layer_idx] = E_to_store
                else:
                    self.E_layer_wise[layer_idx] = torch.cat([prev, E_to_store], dim=1)  # [d, K*t]

                # cap the columns to avoid unbounded growth
                K = len(all_client_ids)
                cap = self.max_bank_rounds * K
                if self.E_layer_wise[layer_idx].shape[1] > cap:
                    self.E_layer_wise[layer_idx] = self.E_layer_wise[layer_idx][:, -cap:]
            else:
                # keep only the latest round’s K residual columns
                self.E_layer_wise[layer_idx] = E_to_store

            # explained energy by top-1 (or r you use for basis)
            try:
                expl = basis_explained_energy({layer_idx: self.E_layer_wise[layer_idx]}, r=self.r_y if hasattr(self, "r_y") else 1)
                explained = float(expl.get(layer_idx, 0.0))
            except Exception:
                explained = 0.0

            # 4) OPTIONAL: smooth across rounds (EMA) or your alpha_mix on E
            # Reuse your prev_spec_global mechanism to stabilize the idiosyncratic residual if you like:
            # E_blend = self.alpha_mix(E, layer_idx, alpha=0.8)  # shape [d,K]
            # For now keep it simple:
            E_blend = E
            
            # 4.1) (Recommended) Bypass decomposition for output head in early rounds
            if server_round <= 5 and layer_idx == self.output_layer_idx:
                # pure FedAvg for head in warmup
                inv_agg = param_matrix @ w                      # [d], weighted mean
                E_blend = param_matrix - inv_agg[:, None]   # residual wrt mean, but we won't scale it below if gamma==1

            # 4.2) Anneal residual contribution sent back to clients
            gamma = self.residual_gamma(server_round)   # e.g., 1.0→0.75→0.5→0.25→0.0 across rounds

            # 5) reconstruct each client's layer: shared mean + its own residual
            for j, cid in enumerate(all_client_ids):
                new_layer_flat = inv_agg + gamma*E_blend[:, j]                 # [d]
                reference_layer = client_params_dict[cid][layer_idx]     # List[Tensor] (shapes)
                new_layer_reshaped = reconstruct_layer_from_flat(new_layer_flat, reference_layer)
                updated_client_params[cid].append(new_layer_reshaped)

            # 6) logging (reusing your CSV format; adapt keys that changed names)
            self.log_server_metrics({
                "round": server_round,
                "layer_idx": layer_idx,
                "inv_variance": float(M_inv.var(dim=1).mean().item()),
                "spec_l1_norm": float(E.abs().mean().item()),         # not exactly L1 pen but useful
                "recon_loss": float(metric_dict["recon_loss"]),
                "inv_norm": float(metric_dict["inv_norm"]),
                "spec_norm": float(metric_dict["idios_norm"]),        # residual norm ~ spec-like
                "agg_norm": float(inv_agg.norm().item()),
                "param_diversity": float(torch.std(param_matrix, dim=1).mean().item()),
                "probe_accuracy_inv": float(metric_dict["probe_accuracy_inv"]),
                "probe_accuracy_spec": float(metric_dict["probe_accuracy_spec"]),
                "cos_to_mean": float(metric_dict["cos_to_mean"]),
                "spread": float(metric_dict["spread"]),
                "angle_inv": float(metric_dict["angle_inv"]),
                "pc1_coords": json.dumps(metric_dict["pc1_coords"]),
                "E_basis_energy": explained
            })
            # end of your server_round function (after logging)
            
        # self.adapt_alpha_beta()
        return updated_client_params












# ======================================================================
# OPTIONAL UTILITIES (NOT WIRED IN YET)
# ----------------------------------------------------------------------
# A) Re-basin / Weight Matching (channel alignment via assignment)
# B) Anchor Projection (site/BN anchors → site/style component)
#
# WHERE TO CALL (later, when you want to use them):
#   • Re-basin: Inside `server_round`, before you call `gather_client_params`,
#     first align each client's layer tensors to a chosen reference model.
#     See the "HOW TO INTEGRATE LATER" notes at the bottom of this block.
#
#   • Anchors: Inside `server_round`, after you build `param_matrix` (Θ ∈ ℝ^{d×K}),
#     you can compute A (K×r) and call `project_onto_anchors(Θ, A)` to peel off
#     a site/style part before your low-rank split. Keep it commented for now.
# ======================================================================

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional

# ------------------------------------------------------------
# (A) RE-BASIN / WEIGHT MATCHING (OPTIONAL)
# ------------------------------------------------------------
try:
    # Hungarian solver for optimal assignment
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _detect_layer_kind(tensors: List[torch.Tensor]) -> str:
    """
    Best-effort guess of layer type using tensor shapes.
    Returns: 'conv', 'linear', or 'other'
    """
    weight = None
    for t in tensors:
        if t.dim() >= 2:
            weight = t
            break
    if weight is None:
        return 'other'
    if weight.dim() in (4, 5):   # Conv2d/Conv3d weights: [Cout, Cin, k*, k* (,k*)]
        return 'conv'
    if weight.dim() == 2:        # Linear weights: [out_features, in_features]
        return 'linear'
    return 'other'


def _find_weight_and_bias(tensors: List[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns (weight, bias) if present in the given 'layer tensors' list.
    """
    weight, bias = None, None
    for t in tensors:
        if t.dim() >= 2 and weight is None:
            weight = t
        elif t.dim() == 1 and bias is None:
            bias = t
    return weight, bias


def _channel_features_from_weight(weight: torch.Tensor) -> torch.Tensor:
    """
    Flattens each OUTPUT channel's parameters into a feature vector.
    - Conv:  weight shape [Cout, Cin, k*, ...] → features shape [Cout, Cin*prod(k*)]
    - Linear: weight shape [Cout, Cin]         → features shape [Cout, Cin]
    """
    assert weight.dim() >= 2
    Cout = weight.shape[0]
    feats = weight.reshape(Cout, -1).contiguous()  # [Cout, Cin * prod(kernel...)]
    return feats


def _pairwise_l2_cost(ref_feats: torch.Tensor, cli_feats: torch.Tensor) -> torch.Tensor:
    """
    Returns a [Cout_ref, Cout_cli] cost matrix of squared L2 distances.
    """
    # (a - b)^2 = a^2 + b^2 - 2ab
    a2 = (ref_feats**2).sum(dim=1, keepdim=True)   # [Cout_ref, 1]
    b2 = (cli_feats**2).sum(dim=1, keepdim=True).t()  # [1, Cout_cli]
    ab = ref_feats @ cli_feats.t()                 # [Cout_ref, Cout_cli]
    cost = (a2 + b2 - 2.0 * ab).clamp_min(0)
    return cost


def _solve_assignment(cost: torch.Tensor) -> torch.Tensor:
    """
    Solves min-cost matching. Returns a permutation index tensor `perm` of length Cout.
    perm[i] = which client channel maps to reference channel i.
    """
    Cout_ref, Cout_cli = cost.shape
    assert Cout_ref == Cout_cli, "Re-basin expects equal #out-channels for ref and client."
    if _HAS_SCIPY:
        # SciPy Hungarian
        row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
        # We expect row_ind == range(Cout); just ensure indexing
        perm = torch.tensor(col_ind, dtype=torch.long, device=cost.device)
    else:
        # Greedy fallback (not optimal but works reasonably)
        perm = torch.empty(Cout_ref, dtype=torch.long, device=cost.device)
        used = torch.zeros(Cout_cli, dtype=torch.bool, device=cost.device)
        # assign in order of lowest row mins
        for i in range(Cout_ref):
            row = cost[i].clone()
            row[used] = float('inf')
            j = int(torch.argmin(row).item())
            perm[i] = j
            used[j] = True
    return perm


def _apply_output_perm_to_layer(tensors: List[torch.Tensor], perm: torch.Tensor) -> List[torch.Tensor]:
    """
    Applies permutation `perm` to OUTPUT channels of a layer (conv/linear).
    - weight Cout dimension = 0
    - bias   Cout dimension = 0
    """
    weight, bias = _find_weight_and_bias(tensors)
    out = []
    for t in tensors:
        if t is weight:
            out.append(torch.index_select(weight, dim=0, index=perm))
        elif t is bias and bias is not None:
            out.append(torch.index_select(bias, dim=0, index=perm))
        else:
            out.append(t)
    return out


def _apply_input_perm_to_layer(tensors: List[torch.Tensor], inv_perm: torch.Tensor) -> List[torch.Tensor]:
    """
    Applies permutation to INPUT channels of a layer (conv/linear).
    - weight Cin dimension = 1 (for conv & linear)
    Other tensors are left untouched.
    """
    weight, bias = _find_weight_and_bias(tensors)
    out = []
    for t in tensors:
        if t is weight:
            out.append(torch.index_select(weight, dim=1, index=inv_perm))
        else:
            out.append(t)
    return out


def match_model_to_reference(
    client_layers: List[List[torch.Tensor]],
    ref_layers: List[List[torch.Tensor]],
    safe_mode: bool = True
) -> List[List[torch.Tensor]]:
    """
    Re-basins a client's layer tensors to align with a reference model.
    Assumes same architecture and layer ordering as your `get_layer_params_list`.

    Parameters
    ----------
    client_layers : list over layers; each item is a list[Tensor] (e.g., [weight, bias, ...])
    ref_layers    : same structure, from the chosen reference model
    safe_mode     : if True, only permutes when shapes are consistent

    Returns
    -------
    aligned_layers : same structure as input; tensors are permuted copies
    """
    aligned = [ [t.clone() for t in layer] for layer in client_layers ]
    L = len(client_layers)

    # Track the output-channel permutation of the previous layer (to fix next layer's inputs)
    for ℓ in range(L):
        kind = _detect_layer_kind(ref_layers[ℓ])
        if kind not in ('conv', 'linear'):
            continue  # skip non-parametric or unsupported layers

        ref_w, _ = _find_weight_and_bias(ref_layers[ℓ])
        cli_w, _ = _find_weight_and_bias(aligned[ℓ])
        if ref_w is None or cli_w is None:
            continue

        # Shapes must match on Cout
        if safe_mode and ref_w.shape[0] != cli_w.shape[0]:
            continue

        # Build cost between OUTPUT channels
        ref_feats = _channel_features_from_weight(ref_w)
        cli_feats = _channel_features_from_weight(cli_w)
        cost = _pairwise_l2_cost(ref_feats, cli_feats)
        perm = _solve_assignment(cost)         # maps ref idx → client idx
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)

        # Apply to CURRENT layer outputs
        aligned[ℓ] = _apply_output_perm_to_layer(aligned[ℓ], perm)

        # Try to apply to NEXT layer inputs (if consistent)
        if ℓ + 1 < L:
            next_kind = _detect_layer_kind(ref_layers[ℓ + 1])
            if next_kind in ('conv', 'linear'):
                nxt_w_ref, _ = _find_weight_and_bias(ref_layers[ℓ + 1])
                nxt_w_cli, _ = _find_weight_and_bias(aligned[ℓ + 1])
                if (nxt_w_ref is not None and nxt_w_cli is not None
                    and nxt_w_ref.dim() >= 2 and nxt_w_cli.dim() >= 2):
                    # Check Cin dimension compatibility
                    cin_ref, cin_cli = nxt_w_ref.shape[1], nxt_w_cli.shape[1]
                    if (not safe_mode) or (cin_ref == cin_cli == perm.numel()):
                        aligned[ℓ + 1] = _apply_input_perm_to_layer(aligned[ℓ + 1], inv_perm)
                    # else: skip (e.g., flatten between conv→linear, skip connections, etc.)

    return aligned


# ------------------------------------------------------------
# (B) ANCHOR PROJECTION (OPTIONAL)
# ------------------------------------------------------------
def build_site_onehot_anchor(K: int, device=None, dtype=None) -> torch.Tensor:
    """
    Returns A ∈ ℝ^{K×K} with site one-hot columns (identity).
    Use as your minimal anchor when BN stats are unavailable.
    """
    A = torch.eye(K, device=device, dtype=dtype)
    return A


def build_bn_anchor_from_vectors(
    bn_vectors: List[torch.Tensor],
    r_pcs: int = 1
) -> torch.Tensor:
    """
    bn_vectors: length-K list; each is a 1D tensor of BN summaries for a given client
                (e.g., concatenated running_mean || running_var for THE SAME LAYER)
    Returns:
        A_pca ∈ ℝ^{K×r_pcs} containing the top r_pcs PCA scores across clients.
    Notes:
        • ONLY use BN from the SAME layer you are decomposing.
        • If you don't have BN running stats in your param lists,
          you'll need to provide bn_vectors via a custom extractor.
    """
    if len(bn_vectors) == 0 or r_pcs < 1:
        raise ValueError("Need at least 1 BN vector and r_pcs >= 1.")
    K = len(bn_vectors)
    with torch.no_grad():
        # Stack to [K, d_bn] on CPU for PCA (sklearn) or fallback SVD
        X = torch.stack([v.detach().flatten().cpu() for v in bn_vectors], dim=0).numpy()  # [K, d_bn]
        Xc = X - X.mean(axis=0, keepdims=True)
        # Safe small-K SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # U:[K,K], S:[K], Vt:[K,d_bn]
        r = min(r_pcs, U.shape[1])
        scores = U[:, :r] * S[:r][None, :]                 # [K, r]
        A_pca = torch.tensor(scores, dtype=bn_vectors[0].dtype, device=bn_vectors[0].device)
    return A_pca


def project_onto_anchors(
    Theta: torch.Tensor,    # [d, K]
    A: torch.Tensor,        # [K, r]
    ridge: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects Θ onto span(A) along the client dimension to extract site/style part.
      Θ_S = Θ Π_A,  Π_A = A (AᵀA + λI)^{-1} Aᵀ
    Returns:
      Theta_S : [d, K]  (anchor-explained portion = site/style)
      R       : [d, K]  (residual = Θ - Θ_S)
    """
    # (AᵀA + λI)^{-1}
    AtA = A.T @ A
    r = AtA.shape[0]
    reg = ridge * torch.eye(r, dtype=AtA.dtype, device=AtA.device)
    inv = torch.inverse(AtA + reg)
    Pi = A @ inv @ A.T      # [K, K]
    Theta_S = Theta @ Pi
    R = Theta - Theta_S
    return Theta_S, R


# ------------------------------------------------------------
# HOW TO INTEGRATE LATER (do NOT enable yet)
# ------------------------------------------------------------
# 1) Re-basin:
#    In Server.server_round(), right after you receive client_params_dict and BEFORE
#    calling gather_client_params() for any layer, choose a reference model's layer list,
#    e.g., the previous global model's params list or the first client as reference.
#
#       ref_layers = client_params_dict[ref_cid]   # list of per-layer tensors (same structure)
#       aligned_client_params_dict = {}
#       for cid, layer_list in client_params_dict.items():
#           if cid == ref_cid:
#               aligned_client_params_dict[cid] = layer_list
#           else:
#               aligned_client_params_dict[cid] = match_model_to_reference(layer_list, ref_layers)
#
#    Then use `aligned_client_params_dict` instead of `client_params_dict` in the rest of the round.
#
#    NOTES:
#    • This will permute ONLY conv/linear outputs and attempt to permute the next layer inputs.
#      If there's a flatten or skip-connection, input-permutation may be skipped automatically.
#    • You should keep the SAME reference within a round. Across rounds, using the previous
#      global model as reference typically stabilizes subspaces.
#
# 2) Anchors (site/BN):
#    In Server.server_round(), after you build param_matrix for a given layer:
#
#       # Suppose K clients; build a site one-hot anchor
#       K = len(client_params_dict)
#       A_onehot = build_site_onehot_anchor(K, device=param_matrix.device, dtype=param_matrix.dtype)
#
#       # OPTIONAL: If you can extract BN running stats per client for THIS layer:
#       # bn_vectors = [bn_extractor(cid, layer_idx) for cid in all_client_ids]  # user-provided
#       # A_bn = build_bn_anchor_from_vectors(bn_vectors, r_pcs=1)
#       # A = torch.cat([A_onehot, A_bn], dim=1)   # [K, r_total]
#       # else:
#       A = A_onehot
#
#       Theta_S, R = project_onto_anchors(param_matrix, A, ridge=1e-3)
#
#    Then, if you want to follow the "peel site/style first" recipe:
#       • Treat Theta_S as M_s (site/style; keep local later),
#       • Run your existing low-rank split on R (residual) instead of Θ.
#    For now, keep this entire anchor section commented out until you're ready.
#
# 3) BN extractor (if/when needed):
#    You'll need to pass BN running_mean / running_var for the SAME layer that produced Θ.
#    Depending on how `get_layer_params_list` groups tensors, BN may be in a separate layer.
#    In that case, write a small function `bn_extractor(cid, layer_idx)` that returns
#    a 1D tensor like torch.cat([running_mean, running_var]), and ensure you call it
#    with the correct BN layer index that corresponds to the weights in Θ.
# ------------------------------------------------------------
