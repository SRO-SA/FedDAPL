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
import json
import pandas as pd
import torch.nn.functional as F
from server import Server


class ServerDomainSpecHelper:
    def __init__(self, server_obj: Server):
        self.server = server_obj
    
    def get_best_layer_idx(self):
        """
        Returns the index of the layer with the best validation score.
        """
        srv = pd.read_csv(self.server.server_log_path)
        layer_scores = {}
        for layer_idx in range(self.server.num_layers):
            layer_data = srv[srv.layer_idx == layer_idx]
            spread_avg = layer_data['spread'].mean()
            angle_inv_avg = layer_data['angle_inv'].mean()
            layer_scores[layer_idx] = spread_avg * angle_inv_avg

        best_layer = max(layer_scores, key=layer_scores.get)
        best_score = layer_scores[best_layer]
        return best_layer, best_score

    
    @debug_function(context="SERVER DOMAIN SPEC")
    def convert_M_spec_layers_to_clients(self, M_spec_dict):
        """
        Convert {layer_idx: M_spec_l [d_l, K]} → list of client vectors [M_spec_client1, ..., M_spec_clientK]
        """
        num_clients = self.server.num_clients
        client_specs = []

        for client_idx in range(num_clients):
            client_vector_parts = []
            for layer_idx in sorted(M_spec_dict.keys()):
                M_spec_layer = M_spec_dict[layer_idx]  # shape [d_l, K]
                M_spec_client_l = M_spec_layer[:, client_idx]  # shape [d_l]
                client_vector_parts.append(M_spec_client_l)
            # Flatten across all layers
            M_spec_client = torch.cat(client_vector_parts)
            client_specs.append(M_spec_client)

        return client_specs
    
    @debug_function(context="SERVER DOMAIN SPEC")
    def compute_spec_distribution(self, M_spec_list):
        """
        M_spec_list: List of domain-specific parameter tensors from trained domains.
                    Each tensor shape: [d] (flattened)
        """
        stacked_spec = torch.stack(M_spec_list)  # shape: [num_domains, d]
        spec_mean = stacked_spec.mean(dim=0)
        spec_std = stacked_spec.std(dim=0) + 1e-6  # small epsilon to avoid zero variance
        return spec_mean, spec_std
        # Example usage after federated training:
        # M_spec_list = [M_spec_client1.flatten(), M_spec_client2.flatten(), ..., M_spec_clientK.flatten()]
        # spec_mean, spec_std = compute_spec_distribution(M_spec_list)
    
    @debug_function(context="SERVER DOMAIN SPEC")
    def initialize_new_domain_spec(self, spec_mean, spec_std):
        """
        Initialize new domain-specific parameters from learned distribution.
        """
        new_spec = torch.normal(mean=spec_mean, std=spec_std)
        return new_spec

        # Usage:
        # new_M_spec = initialize_new_domain_spec(spec_mean, spec_std)
        # print(new_M_spec.shape)  # [d]
        
    # @torch.no_grad()
    # def sample_mahalanobis(self, M_spec_list, epsilon=1.0):
    #     """
    #     Return a brand-new spec that is ε-far (Mahalanobis) from the mean,
    #     but still lies in the training ellipsoid.
    #     """
    #     X   = torch.stack(M_spec_list)              # (K, d)
    #     mu  = X.mean(0)
    #     cov = torch.cov(X.T) + 1e-6*torch.eye(X.size(1), device=X.device)
    #     L   = torch.linalg.cholesky(cov)

    #     unit = torch.randn_like(mu)
    #     unit = unit / unit.norm()                   # random direction
    #     new  = mu + L @ unit * epsilon
    #     return new
    
    # @torch.no_grad()
    # def sample_mahalanobis_lowrank(self, M_spec_list, epsilon=1.0):
    #     """
    #     Low-rank Mahalanobis sampler (rank ≤ K-1).
    #     spec_list : list of K tensors, each shape (d,)
    #     epsilon   : radius multiplier (0.3–0.6 recommended)

    #     Returns
    #     -------
    #     new_spec  : tensor (d,)
    #     """
    #     X = torch.stack(M_spec_list)           # (K, d)
    #     mu = X.mean(0)
    #     Xc = (X - mu).T                      # (d, K)
    #     K  = Xc.shape[1]

    #     G  = (Xc.T @ Xc) / (K-1)             # (K, K)  tiny
    #     eigval, V = torch.linalg.eigh(G)     # eigval ascending
    #     eigval = eigval.clamp_min(1e-9)      # numerical safety
    #     # Lam_inv_sqrt = torch.diag(eigval.rsqrt())
    #     Lam_sqrt  = torch.diag(eigval.sqrt())  # √Λ   <-- **square-root**, not inverse

    #     # random direction on K-sphere
    #     w = torch.randn(K, device=X.device)
    #     w = epsilon * w / w.norm()

    #     delta = (Xc @ (V @ (Lam_sqrt @ w))) / math.sqrt(K - 1)
    #     new = mu + delta
    #     return new
    
    # @torch.no_grad()
    # def sample_lowrank_clip(self, M_spec_list, frac=0.6):
    #     new = self.sample_mahalanobis_lowrank(M_spec_list, epsilon=1.0)  # any ε
    #     mu  = torch.stack(M_spec_list).mean(0)
    #     Xc  = torch.stack(M_spec_list) - mu
    #     rms = torch.sqrt((Xc**2).sum() / (len(M_spec_list)-1)).item()  # √trace
    #     delta = new - mu
    #     target = frac * rms                        # e.g. 0.6× training rms
    #     new = mu + delta * (target / delta.norm()) # clip radius
    #     return new
    
    # @torch.no_grad()
    # def sample_mahalanobis_diag(self, M_spec_list, epsilon=1.0):
    #     X   = torch.stack(M_spec_list)          # (K, d)
    #     mu  = X.mean(0)
    #     std = X.std(0) + 1e-6                # diag Σ½
    #     z   = torch.randn_like(mu)
    #     z   = z / z.norm()                   # unit direction
    #     new = mu + epsilon * std * z         # ε-far in diag metric
    #     return new
    
    # @torch.no_grad()
    # def sample_dirichlet(self, spec_list, alpha=0.3):
    #     K = len(spec_list)
    #     w = torch.distributions.dirichlet.Dirichlet(alpha * torch.ones(K)).sample()
    #     new = torch.stack(spec_list).T @ w        # convex combo
    #     return new
    
    # @torch.no_grad()
    # def sample_layer4_pc(self, spec_list, frac=0.4):
    #     """
    #     Move along the top-1 PC of layer-4 (index=4) by
    #     `frac` × training std in that direction.
    #     """
    #     layer4 = self.server.layer_slices[4]
    #     # stack only the layer-4 part of every client spec (all rounds)
    #     X = torch.stack([s[layer4] for s in spec_list])      # (K, d4)
    #     mu4 = X.mean(0)
    #     # PCA on CPU numpy
    #     pc1 = PCA(n_components=1).fit(X.cpu().numpy()).components_[0]
    #     pc1 = torch.tensor(pc1, device=X.device, dtype=X.dtype)
    #     pc1 = pc1 / pc1.norm()                               # unit vector

    #     std1 = X.sub(mu4).matmul(pc1).std()                  # std along PC-1

    #     new_flat = mu4 + frac * std1 * pc1                   # move 0.4·σ
    #     # --------------------------------------------------------------
    #     # Build full spec vector: layer-4 gets new_flat, others = mean
    #     # --------------------------------------------------------------
    #     spec_mean_flat = torch.stack(spec_list).mean(0)      # (d,)
    #     out = spec_mean_flat.clone()
    #     out[layer4] = new_flat
    #     return out
    
    # @torch.no_grad()
    # def sample_bestlayer_dirichlet(self, spec_list, layer_idx, alpha=[1.5,1.5,1.5]):
    #     layer_slice = self.server.layer_slices[layer_idx]
    #     X = torch.stack([s[layer_slice] for s in spec_list])
    #     mu = X.mean(0)

    #     # Low‐rank SVD to get U ∈ ℝ^(K×1), S ∈ ℝ^1, V ∈ ℝ^(d_l×1)
    #     U, S, V = torch.svd_lowrank(X - mu[None], q=1)
    #     # U[:,0] are the PC1 *scores* for each of the K clients
    #     pc1 = U[:, 0]                             # length K

    #     pc1_dist = torch.abs(pc1 - pc1.mean()).abs()
    #     idx_top3 = pc1_dist.topk(3, largest=False).indices  # tensor length 3


    #     w = torch.distributions.Dirichlet(torch.tensor(alpha)).sample().to(X.device)

    #     new_flat = sum(w[i] * spec_list[int(idx_top3[i])] for i in range(3))

    #     return new_flat

    
    def layer_grad_norms(self, model, val_loader, layer_param_groups,
                        n_batches=2, device="cpu"):
        if device is None:
            device = next(model.parameters()).device
        model.to(device)
        norms = [0.0] * len(layer_param_groups)
        criterion = torch.nn.MSELoss()

        it = iter(val_loader)
        for _ in range(n_batches):
            try:
                xb, yb = next(it)[:2]           # works for (x,y) or (x,y,meta)
            except StopIteration:
                break

            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb.float()), yb.squeeze(1))

            model.zero_grad(set_to_none=True)
            loss.backward()

            # accumulate grad-norms per layer
            for j, plist in enumerate(layer_param_groups):
                layer_norm_sq = 0.0
                for p in plist:
                    if p.grad is not None:
                        layer_norm_sq += p.grad.norm().item() ** 2
                norms[j] += math.sqrt(layer_norm_sq)

        return [n / max(1, n_batches) for n in norms]


    # ==========================================================
    # UNSEEN DOMAIN HELPERS (NOT CALLED YET)
    # ==========================================================
    import torch
    import numpy as np
    from typing import Dict, List, Optional

    # ---------- (A) Compose a model for an unseen domain ----------
    def compose_unseen_from_shared_and_E(self, shared_mean_layers: Dict[int, torch.Tensor],
                                        E_choice_layers: Dict[int, torch.Tensor],
                                        reference_struct: Dict[int, List[torch.Tensor]]):
        """
        Build structured per-layer tensors for an unseen domain:
        theta_unseen[ℓ] = shared_mean_layers[ℓ] + E_choice_layers[ℓ]
        Then reshape using `reference_struct[ℓ]` (list of tensors' shapes).
        Returns: {layer_idx: [Tensor,...]} shaped like your model for set_state_dict.
        
        WHERE TO USE:
        • After training, on the machine doing validation/inference.
        • Provide `reference_struct` from any client's layer structure or a dummy model.
        """
        out = {}
        for ℓ, shared_flat in shared_mean_layers.items():
            e_flat = E_choice_layers.get(ℓ, torch.zeros_like(shared_flat))
            flat = (shared_flat + e_flat).to(shared_flat.device)
            ref = reference_struct[ℓ]  # list[Tensor] with shapes
            # Reuse your util reconstruct_layer_from_flat if in scope
            out[ℓ] = reconstruct_layer_from_flat(flat, ref)
        return out


    # ---------- (B) Simple residual pickers / samplers ----------
    def pick_zero_residual(self, shared_mean_layers: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """E=0 per layer."""
        return {ℓ: torch.zeros_like(v) for ℓ, v in shared_mean_layers.items()}


    def pick_nearest_by_bn_anchor(self, residual_bank: Dict[int, torch.Tensor],
                                target_bn_anchor: Dict[int, torch.Tensor],
                                train_bn_anchors: Dict[int, List[torch.Tensor]]):
        """
        Choose the residual column whose BN anchor is closest to the target (per layer).
        `residual_bank[ℓ]` is [d, K]; `train_bn_anchors[ℓ]` is list length K of BN vectors (same layer).
        `target_bn_anchor[ℓ]` is a 1D BN vector computed on a small unlabeled target batch.
        
        WHERE TO USE:
        • If you can run an unlabeled batch from the unseen site locally to collect BN stats.
        • You must have saved per-layer BN vectors for training sites (not included here).
        """
        E_choice = {}
        for ℓ, E in residual_bank.items():
            K = E.shape[1]
            # Cosine distance to target BN vector
            t = target_bn_anchor[ℓ] / (target_bn_anchor[ℓ].norm() + 1e-8)
            sims = []
            for k in range(K):
                a = train_bn_anchors[ℓ][k]
                a = a / (a.norm() + 1e-8)
                sims.append((a * t).sum().item())  # cosine similarity
            j = int(np.argmax(sims))
            E_choice[ℓ] = E[:, j]
        return E_choice


    def sample_residual_dirichlet(self, residual_bank: Dict[int, torch.Tensor], alpha: float = 0.3):
        """
        Sample a convex combination of residual columns per layer (Dirichlet).
        """
        E_choice = {}
        for ℓ, E in residual_bank.items():
            d, K = E.shape
            w = torch.distributions.dirichlet.Dirichlet(alpha * torch.ones(K)).sample().to(E.device)
            E_choice[ℓ] = (E @ w)  # [d]
        return E_choice


    def sample_residual_pc(self, residual_bank: Dict[int, torch.Tensor], frac: float = 0.4):
        """
        Move along top-1 PC of residuals per layer by `frac` × std in that direction.
        """
        E_choice = {}
        for ℓ, E in residual_bank.items():
            X = E - E.mean(dim=1, keepdim=True)  # center across clients
            # Small-SVD in client space (rank ≤ K)
            U, S, V = torch.svd_lowrank(X, q=1)  # V: [K,1] are client scores; U: [d,1]
            std1 = S[0] / (X.shape[1] ** 0.5 + 1e-8)
            u1 = U[:, 0]
            E_choice[ℓ] = frac * std1 * u1       # signed is arbitrary → zero-mean perturb
        return E_choice


    def sample_residual_lowrank_mahalanobis(self, residual_bank: Dict[int, torch.Tensor],
                                            epsilon: float = 1.0):
        """
        Low-rank Mahalanobis draw in the (≤K)-dim client subspace spanned by residuals.
        """
        E_choice = {}
        for ℓ, E in residual_bank.items():
            # Work in the K-dim column space: E = B * C, with C:[r,K]
            X = E - E.mean(dim=1, keepdim=True)   # [d,K]
            U, S, V = torch.svd_lowrank(X, q=min(3, X.shape[1]))  # keep ≤3 comps
            # Sample on r-sphere
            r = U.shape[1]
            if r == 0:
                E_choice[ℓ] = torch.zeros(E.shape[0], dtype=E.dtype, device=E.device)
                continue
            z = torch.randn(r, device=E.device)
            z = epsilon * z / (z.norm() + 1e-8)
            # Map back to weight space
            E_choice[ℓ] = (U @ (S[:r] * z))
        return E_choice


    # ---------- (C) Tiny BN adaptation (optional, unlabeled) ----------
    def bn_moment_adapt(self, model, data_loader, n_batches: int = 10, device="cuda"):
        """
        Update BN running_mean/var on a small unlabeled batch from the unseen domain.
        No gradients. Improves scanner shift often.
        
        WHERE TO USE:
        • On the validation/inference machine, after composing the unseen model.
        """
        was_training = model.training
        model.train()  # to update BN buffers
        cnt = 0
        with torch.no_grad():
            for xb, *_ in data_loader:
                xb = xb.to(device).float()
                _ = model(xb)
                cnt += 1
                if cnt >= n_batches:
                    break
        model.train(was_training)


