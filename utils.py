
import torch
import logging
import os
import time
from functools import wraps
from colorama import init, Fore, Style
import numpy as np
import random
from pathlib import Path
# utils.py
import contextlib
from typing import Dict, List, Tuple, Optional, Any
import torch.nn.functional as F

init(autoreset=True)

# Create logs folder
os.makedirs("logs", exist_ok=True)

# Logger setup
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Console handler with color
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler (no color)
sty_env = os.environ.get("STY", None)
if sty_env:
    session_name = sty_env.split(".")[-1]  # Extract after the dot
else:
    session_name = "no_screen"

slurm_job_id = os.environ.get("SLURM_JOB_ID", "nojob")
slurm_proc_id = os.environ.get("SLURM_PROCID", "noproc")
log_file_name = f"logs/debug_log_{session_name}_{slurm_job_id}_{slurm_proc_id}.txt"

file_handler = logging.FileHandler(log_file_name, mode="w")
file_handler.setLevel(logging.DEBUG)

# Formatter
file_format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
console_format = logging.Formatter("%(message)s")  # We format manually below

file_handler.setFormatter(file_format)
console_handler.setFormatter(console_format)

if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

DEBUG = True

# Color helpers
def blue(text): return Fore.CYAN + text + Style.RESET_ALL
def yellow(text): return Fore.YELLOW + text + Style.RESET_ALL
def green(text): return Fore.GREEN + text + Style.RESET_ALL
def magenta(text): return Fore.MAGENTA + text + Style.RESET_ALL
def bold(text): return Style.BRIGHT + text + Style.RESET_ALL

# Label helpers
def tag(context): return green(f"[{context.upper()}]")

def get_layer_params_list(model, clone=True):
    """
    Convert layer-param dict to list of lists (for index-based layer access).
    """
    layer_dict = get_layer_params_dict(model, clone)
    return list(layer_dict.values())

def get_layer_params_dict(model, clone=True):
    layer_param_dict = {}
    for name, param in model.named_parameters():
        layer = name.split(".")[0]
        if layer not in layer_param_dict:
            layer_param_dict[layer] = []
        layer_param_dict[layer].append(
            param.clone().detach() if clone else param
        )
    return layer_param_dict

def flatten_layer_param_list_for_flower(layer_param_list):
    """
    Accepts either the usual List[List[Tensor]] *or* an nn.Module.
    In both cases it returns a flat list ordered by
    `model.named_parameters()`, which is identical on the server
    and on every client.
    """
    import itertools
    # --- case 1: we were given Model.get_params_for_layers() ------------
    if isinstance(layer_param_list, (list, tuple)):
        flat = []
        for t in itertools.chain.from_iterable(layer_param_list):
            flat.append(t.detach().cpu().numpy())
        return flat

    # --- case 2: we were (accidentally) passed the model itself ---------
    if isinstance(layer_param_list, torch.nn.Module):
        flat = [p.detach().cpu().numpy()
                for _, p in layer_param_list.named_parameters()]
        return flat

    raise TypeError("Expected list-of-layers or nn.Module, "
                    f"got {type(layer_param_list)}")
    
    # flat = []
    # for layer in layer_param_list:
    #     for p in layer:
    #         np_p = p.detach().cpu().numpy()
    #         if np_p.shape == ():  # if scalar, this is a problem
    #             raise ValueError("Scalar tensor detected during flattening. This shouldn't happen.")
    #         flat.append(np_p)
    # return flat

def flatten_layer_param_list_for_model(layer_param_list):
    # return [torch.tensor(p) for layer in layer_param_list for p in layer]
    return [
        p.clone().detach() if isinstance(p, torch.Tensor) else torch.tensor(p)
        for layer in layer_param_list for p in layer
    ]

def reconstruct_layer_param_list(flat_params, reference_structure):
    reconstructed = []
    idx = 0
    for layer in reference_structure:
        param_count = len(layer)
        new_layer = [p if isinstance(p, torch.Tensor) else torch.tensor(p) for p in flat_params[idx:idx + param_count]]
        reconstructed.append(new_layer)
        idx += param_count
    return reconstructed

def reconstruct_layer_from_flat(flat_tensor, reference_layer):
    """
    Takes a flattened 1D tensor and a reference [Tensor1, Tensor2, ...]
    and reshapes chunks of the flat tensor to match each reference tensor.
    Returns: [reshaped_Tensor1, reshaped_Tensor2, ...]
    """
    new_params = []
    idx = 0
    for ref_tensor in reference_layer:
        numel = ref_tensor.numel()
        reshaped = flat_tensor[idx:idx+numel].view_as(ref_tensor)
        new_params.append(reshaped)
        idx += numel
    return new_params


from flwr.common import Parameters, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy

def is_large_flower_object(obj):
    large_types = (Parameters, ClientProxy, FitRes, EvaluateRes)
    
    if isinstance(obj, large_types):
        return True

    # Check if it's a list/tuple of large FL objects
    if isinstance(obj, (list, tuple)):
        if len(obj) > 0 and any(is_large_flower_object(x) for x in obj):
            return True

    # If it's a tuple like (ClientProxy, FitRes)
    if isinstance(obj, tuple) and len(obj) == 2:
        if isinstance(obj[0], ClientProxy) and isinstance(obj[1], (FitRes, EvaluateRes)):
            return True

    return False


# Tensor info logger
def log_tensor_info(name, tensor, context="GENERAL"):
    if not isinstance(tensor, torch.Tensor):
        return
    try:
        logger.debug(
            f"{tag(context)} {magenta(name)} 📦 "
            f"shape: {tuple(tensor.shape)}, device: {tensor.device}, dtype: {tensor.dtype}, "
            f"min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}, mean: {tensor.mean().item():.4f}"
        )
    except Exception as e:
        logger.debug(f"{tag(context)} {name} -> Failed to log tensor info: {e}")

def safe_log_value(name, value, context="GENERAL"):
    try:
        if isinstance(value, torch.Tensor):
            log_tensor_info(name, value, context)

        elif is_large_flower_object(value):
            logger.debug(f"{tag(context)} {yellow(name)} skipped logging large FL object ({type(value).__name__})")

        elif isinstance(value, (int, float, str, bool)):
            logger.debug(f"{tag(context)} {yellow(name)} = {value}")

        elif isinstance(value, (list, tuple, dict, set)):
            if len(value) > 5:
                logger.debug(f"{tag(context)} {yellow(name)} skipped logging large {type(value).__name__} (length={len(value)})")
            else:
                # log elements individually if they're safe
                logger.debug(f"{tag(context)} {yellow(name)} contains {len(value)} items")
                for i, v in enumerate(value):
                    if not is_large_flower_object(v):
                        safe_log_value(f"{name}[{i}]", v, context)
                    else:
                        logger.debug(f"{tag(context)} {yellow(name)}[{i}] skipped logging large item ({type(v).__name__})")
        else:
            logger.debug(f"{tag(context)} {yellow(name)} skipped logging object of type {type(value).__name__}")

    except Exception as e:
        logger.debug(f"{tag(context)} {name} -> Error while logging: {e}")

def debug_function(context="GENERAL"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if DEBUG:
                logger.debug(f"{tag(context)} 🔧 {bold(func.__name__)} called.")
                for i, arg in enumerate(args):
                    safe_log_value(f"arg[{i}]", arg, context)
                for k, v in kwargs.items():
                    safe_log_value(f"kwarg[{k}]", v, context)
                start = time.time()

            result = func(*args, **kwargs)

            if DEBUG:
                duration = time.time() - start
                logger.debug(f"{tag(context)} ✅ {bold(func.__name__)} completed in {duration:.3f}s")

                # Handle Return separately and safely
                if isinstance(result, (list, tuple)) and all(
                    isinstance(item, (tuple, list)) and any(is_large_flower_object(sub) for sub in item)
                    for item in result
                ):
                    logger.debug(f"{tag(context)} Return skipped logging large list of client/proxy-related results")
                else:
                    safe_log_value("Return", result, context)
            return result
        return wrapper
    return decorator


def log_print(*args, level="DEBUG", context="GENERAL", sep=" ", end="\n"):
    """A print-like logger that logs to both file and colorful console."""
    # Combine args into a single string like print does
    message = sep.join(str(arg) for arg in args) + end

    # Plain log to file
    file_message = f"[{context}] {message.strip()}"
    logger.log(getattr(logging, level.upper(), logging.DEBUG), file_message)

    # Colorful console version
    color_tag = Fore.GREEN + f"[{context}]" + Style.RESET_ALL
    console_message = color_tag + " " + message.strip()
    print(console_message)
    
    
# ============================ NEW: E-SCHEDULER HELPERS ============================


class PerRoundEScheduler:
    """
    Samples and applies layer-wise residual deltas E to the model at the start of each epoch.
    Modes:
      - 'off'        : do nothing
      - 'dirichlet'  : convex combo of residual-bank columns per layer
      - 'pc'         : move along top-1 PC of residuals per layer
      - 'lowrank'    : low-rank Mahalanobis sample in residual subspace per layer
    Inputs (optional, via config):
      - residual_bank: dict[int -> Tensor[d, K]]  # per-layer bank from server (heavy)
      - residual_basis: dict[int -> {'U': Tensor[d, r], 'scale': float}]  # light
      - rank, scale, zero_out_last_layer, max_layer_norm
    """
    def __init__(self,
                 model,
                 get_params_for_layers_fn,
                 reconstruct_from_flat_fn,
                 layer_count: int,
                 output_layer_idx: int,
                 mode: str = "off",
                 residual_bank: Optional[Dict[int, torch.Tensor]] = None,
                 residual_basis: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
                 rank: int = 1,
                 scale: float = 0.4,
                 max_layer_norm: Optional[float] = None,
                 zero_out_last_layer: bool = True,
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None):
        self.model = model
        self.get_params_for_layers_fn = get_params_for_layers_fn
        self.reconstruct_from_flat_fn = reconstruct_from_flat_fn
        self.L = layer_count
        self.out_idx = output_layer_idx
        self.mode = mode
        self.bank = residual_bank or {}
        self.basis = residual_basis or {}
        self.rank = rank
        self.scale = scale
        self.max_layer_norm = max_layer_norm
        self.zero_out_last_layer = zero_out_last_layer
        self.device = device or next(model.parameters()).device
        self.rng = np.random.RandomState(seed if seed is not None else 0)
        random.seed(seed if seed is not None else 0)
        # In PerRoundEScheduler.__init__
        self._clip_hits = 0
        self._clip_total = 0

        # Filled by client when round begins
        self.round_start_snapshot: Optional[List[List[torch.Tensor]]] = None

    def set_round_start_snapshot(self, layer_tensors: List[List[torch.Tensor]]):
        # Deep copy tensors (keep on the same device)
        self.round_start_snapshot = [[t.detach().clone() for t in layer] for layer in layer_tensors]

    def active(self) -> bool:
        return self.mode is not None and self.mode.lower() != "off"

    @torch.no_grad()
    def restore_snapshot(self):
        if self.round_start_snapshot is None:
            return
        # Overwrite model with snapshot params
        layers_current = self.get_params_for_layers_fn()
        for layer, snap in zip(layers_current, self.round_start_snapshot):
            for p, q in zip(layer, snap):
                p.data.copy_(q.data)

    @torch.no_grad()
    def apply_delta_layers(self, delta_layers_flat: Dict[int, torch.Tensor]):
        """delta_layers_flat: per-layer flat delta to ADD to current weights."""
        layers_current = self.get_params_for_layers_fn()
        for ℓ, delta_flat in delta_layers_flat.items():
            if delta_flat is None: 
                continue
            if self.zero_out_last_layer and ℓ == self.out_idx:
                continue
            ref = layers_current[ℓ]
            # reconstruct delta tensors with same shapes as ref
            delta_tensors = self.reconstruct_from_flat_fn(delta_flat.to(self.device), ref)
            for p, d in zip(ref, delta_tensors):
                p.data.add_(d)

    # ------------------------ samplers ------------------------
    def _dirichlet_sample(self, E: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
        # E: [d, K]
        d, K = E.shape
        w = torch.distributions.dirichlet.Dirichlet(alpha * torch.ones(K, device=E.device)).sample()
        return E @ w  # [d]

    def _pc_sample(self, E: torch.Tensor, frac: float) -> torch.Tensor:
        # Center across clients then take top-1 SVD
        X = E - E.mean(dim=1, keepdim=True)
        q = min(1, X.shape[1])  # at least 1
        if q < 1 or X.abs().sum() == 0:
            return torch.zeros(E.shape[0], device=E.device, dtype=E.dtype)
        U, S, V = torch.svd_lowrank(X, q=1)
        std1 = S[0] / (X.shape[1] ** 0.5 + 1e-8)
        u1 = U[:, 0]
        return (frac * std1) * u1

    def _lowrank_mahalanobis(self, E: torch.Tensor, epsilon: float) -> torch.Tensor:
        X = E - E.mean(dim=1, keepdim=True)  # [d, K]
        r = min(self.rank, X.shape[1])
        if r < 1 or X.abs().sum() == 0:
            return torch.zeros(E.shape[0], device=E.device, dtype=E.dtype)
        U, S, _ = torch.svd_lowrank(X, q=r)
        z = torch.randn(r, device=E.device)
        z = epsilon * z / (z.norm() + 1e-8)
        return U @ (S[:r] * z)

    def _basis_sample(self, U: torch.Tensor, scale: float, r: int) -> torch.Tensor:
        # U: [d, rU] basis; draw z ~ N(0, I_r) and return scale * U z
        r_use = min(r, U.shape[1])
        if r_use < 1:
            return torch.zeros(U.shape[0], device=U.device, dtype=U.dtype)
        z = torch.randn(r_use, device=U.device)
        z = z / (z.norm() + 1e-8)
        return scale * (U[:, :r_use] @ z)

    def _clip_trust_region(self, delta: torch.Tensor) -> torch.Tensor:
        self._clip_total += 1
        if self.max_layer_norm is not None:
            n = delta.norm()
            if n > self.max_layer_norm:
                self._clip_hits += 1
                return delta * (self.max_layer_norm / (n + 1e-12))
        return delta
    
    @torch.no_grad()
    def sample_delta_for_layer(self, ℓ: int) -> Optional[torch.Tensor]:
        mode = (self.mode or "off").lower()
        # Prefer provided basis if available (smaller payload than full bank)
        if ℓ in self.basis and mode in ("pc", "lowrank", "dirichlet"):
            U = self.basis[ℓ]["U"].to(self.device)
            base = float(self.basis[ℓ].get("scale", 1.0))   # per-layer magnitude from SVD
            global_scale = float(getattr(self, "scale", 1.0))  # from config["E_scale"]
            scale = base * global_scale
            delta = self._basis_sample(U, scale=scale, r=self.rank)
            return self._clip_trust_region(delta)

        if ℓ not in self.bank:
            return None

        E = self.bank[ℓ].to(self.device)  # [d, K]
        if mode == "dirichlet":
            delta = self._dirichlet_sample(E, alpha=0.3)
        elif mode == "pc":
            delta = self._pc_sample(E, frac=self.scale)
        elif mode == "lowrank":
            delta = self._lowrank_mahalanobis(E, epsilon=self.scale)
        else:
            return None

        return self._clip_trust_region(delta)

    @torch.no_grad()
    def build_epoch_delta(self) -> Dict[int, torch.Tensor]:
        deltas = {}
        for ℓ in range(self.L):
            if self.zero_out_last_layer and ℓ == self.out_idx:
                deltas[ℓ] = torch.zeros(0)  # skip
                continue
            dℓ = self.sample_delta_for_layer(ℓ)
            if dℓ is not None:
                deltas[ℓ] = dℓ
        return deltas
# ========================= END: E-SCHEDULER HELPERS ===============================

    def active(self) -> bool:
        return getattr(self, "mode", "off") != "off"

    def _early_layer_filter(self, layer_idx: int, L: int) -> bool:
        # used by dropout; apply to all but the last layer by default
        return layer_idx < (L - 1)

    def _fallback_direction(self, d: int, device, dtype):
        # used if no basis is available
        v = torch.zeros(d, device=device, dtype=dtype)
        v[0] = 1.0
        return v.view(-1, 1)

    def _basis_for_layer(self, layer_idx: int, d: int, device, dtype):
        """Return (U, r) or (None, None). U is [d, r]."""
        B = getattr(self, "basis", None)
        if not B or layer_idx not in B:
            return None, None
        U = B[layer_idx]["U"]
        if isinstance(U, np.ndarray):
            U = torch.from_numpy(U)
        U = U.to(device=device, dtype=dtype)
        if U.dim() == 1:
            U = U.view(-1, 1)
        return U, U.shape[1]

    def project_grad_to_E(self, g: torch.Tensor, layer_idx: int, L_total: int) -> torch.Tensor:
        """Orthogonal projection of grad 'g' onto span(U) if available, else 0."""
        U, _ = self._basis_for_layer(layer_idx, g.numel(), g.device, g.dtype)
        if U is None:
            return torch.zeros_like(g)
        G = U.t() @ U
        G = G + 1e-6 * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        P = U @ torch.linalg.solve(G, U.t())
        return P @ g

    def sample_delta_E(self, flat_layers: list[torch.Tensor]) -> list[torch.Tensor]:
        """Sample a small delta in E for each layer; returns list of flats matching input."""
        deltas = []
        L = len(flat_layers)
        step = float(getattr(self, "scale", 0.4))
        max_norm = getattr(self, "max_layer_norm", None)
        for ℓ, w in enumerate(flat_layers):
            d, device, dtype = w.numel(), w.device, w.dtype
            if getattr(self, "zero_out_last_layer", True) and ℓ == getattr(self, "output_layer_idx", L - 1):
                deltas.append(torch.zeros_like(w)); continue
            U, r = self._basis_for_layer(ℓ, d, device, dtype)
            if U is None or r is None or r < 1:
                deltas.append(torch.zeros_like(w)); continue
            coeff = torch.randn(r, device=device, dtype=dtype)
            coeff = coeff / (coeff.norm() + 1e-12)
            delta = U @ (step * coeff)
            if max_norm is not None:
                n = delta.norm()
                if n > max_norm:
                    delta = delta * (max_norm / (n + 1e-12))
            deltas.append(delta)
        return deltas

    @contextlib.contextmanager
    def apply_delta_temporarily(self, layer_param_groups: list[list[torch.nn.Parameter]],
                                delta_flats: list[torch.Tensor]):
        """
        Apply deltas on top of the CURRENT weights (which may already include your per-epoch delta),
        run a forward/backward, then restore exactly.
        """
        # Save current tensors (not just snapshot) so we compose cleanly with your epoch delta
        saved = [[p.detach().clone() for p in plist] for plist in layer_param_groups]

        # Apply (flat + delta) → reshape → copy back
        for plist, delta in zip(layer_param_groups, delta_flats):
            flat = torch.cat([p.view(-1) for p in plist])
            new_flat = flat + delta
            new_list = self.reconstruct_from_flat_fn(new_flat, plist)
            for p, newp in zip(plist, new_list):
                p.data.copy_(newp)

        try:
            yield
        finally:
            # Restore
            for plist, s in zip(layer_param_groups, saved):
                for p, ps in zip(plist, s):
                    p.data.copy_(ps)


def _mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Works for [B] or [B,1]
    return (pred.view(-1) - target.view(-1)).abs().mean()

def _p95(x: List[float]) -> float:
    if not x:
        return 0.0
    a = np.asarray(x, dtype=np.float64)
    return float(np.percentile(a, 95.0))

def _flatten_layer_group(group: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in group])

def _project_onto_basis(g: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    # P = U (U^T U)^-1 U^T
    G = U.T @ U
    G = G + 1e-6 * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    return U @ torch.linalg.solve(G, U.T @ g)

@contextlib.contextmanager
def apply_flat_deltas_temporarily(
    layer_param_groups: List[List[torch.nn.Parameter]],
    deltas_flat: List[Optional[torch.Tensor]],
    reconstruct_from_flat_fn,
):
    # Save current tensors
    saved = [[p.detach().clone() for p in plist] for plist in layer_param_groups]
    # Apply deltas
    for plist, delta in zip(layer_param_groups, deltas_flat):
        if delta is None:
            continue
        base_flat = _flatten_layer_group(plist)
        new_flat  = base_flat + delta.to(base_flat.device, base_flat.dtype)
        new_list  = reconstruct_from_flat_fn(new_flat, plist)
        for p, newp in zip(plist, new_list):
            p.data.copy_(newp)
    try:
        yield
    finally:
        # Restore
        for plist, s in zip(layer_param_groups, saved):
            for p, ps in zip(plist, s):
                p.data.copy_(ps)

def _p95(vals):
    if not vals:
        return 0.0
    v = np.sort(np.asarray(vals, dtype=np.float64))
    idx = int(0.95 * (len(v) - 1))
    return float(v[idx])

def _project_onto_basis(g: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    # g:[d], U:[d,r]
    if U.dim() == 1:
        U = U.view(-1, 1)
    G = U.t() @ U + 1e-6 * torch.eye(U.shape[1], device=U.device, dtype=U.dtype)
    return U @ torch.linalg.solve(G, U.t() @ g)

def gradE_ratio_for_layers(
    model,
    dataloader,
    basis: Dict[int, Dict[str, torch.Tensor]],
    get_params_for_layers_fn,
    loss_fn = torch.nn.MSELoss(),
    n_batches: int = 1,
    zero_out_last_layer: bool = True,
    out_idx: Optional[int] = None,
) -> Tuple[Dict[int, float], float, float]:
    device = next(model.parameters()).device
    model.train()  # need grads
    ratios_per_layer: Dict[int, List[float]] = {}
    it = iter(dataloader)

    with torch.enable_grad():  # <-- ensure grads are on even if called in a no_grad context
        for _ in range(n_batches):
            try:
                x, y, _meta = next(it)
            except StopIteration:
                break
            x = x.to(device); y = y.to(device)

            model.zero_grad(set_to_none=True)  # <-- safer than manual loop
            pred = model(x.float())
            loss = loss_fn(pred, y)
            loss.backward()

            layer_groups = get_params_for_layers_fn()
            for ℓ, plist in enumerate(layer_groups):
                if zero_out_last_layer and out_idx is not None and ℓ == out_idx:
                    continue

                # collect flat gradient for this layer
                g_list = []
                for p in plist:
                    if p.grad is None:
                        g_list.append(torch.zeros_like(p, device=device).view(-1))
                    else:
                        g_list.append(p.grad.view(-1))
                if not g_list:
                    continue
                g = torch.cat(g_list)
                gnorm = float(g.norm().item())
                if gnorm == 0.0:
                    continue

                if ℓ not in basis:
                    continue
                U = basis[ℓ]["U"].to(device=device, dtype=g.dtype)
                gE = _project_onto_basis(g, U)
                rho = (gE.norm() / (g.norm() + 1e-8)).item()
                ratios_per_layer.setdefault(ℓ, []).append(rho)

    per_layer = {ℓ: float(np.mean(vals)) for ℓ, vals in ratios_per_layer.items()}
    all_vals = [v for vals in ratios_per_layer.values() for v in vals]
    mean_rho = float(np.mean(all_vals)) if all_vals else 0.0
    p95_rho  = _p95(all_vals)
    return per_layer, mean_rho, p95_rho



@torch.no_grad()
def directional_sensitivity_along_basis(
    model,
    dataloader,
    basis: Dict[int, Dict[str, torch.Tensor]],
    get_params_for_layers_fn,
    reconstruct_from_flat_fn,
    eps: float = 1e-3,
    n_batches: int = 1,
    zero_out_last_layer: bool = True,
    out_idx: Optional[int] = None,
) -> Dict[int, float]:
    """
    Returns per-layer sensitivity:
      sens_ℓ = mean_b || f(W) - f(W + eps * U_ℓ[:,0]) || / eps
    """
    device = next(model.parameters()).device
    model.eval()
    it = iter(dataloader)
    sens = {}

    for ℓ, entry in basis.items():
        if zero_out_last_layer and out_idx is not None and ℓ == out_idx:
            continue
        U = entry["U"].to(device)
        if U.numel() == 0:
            continue
        u0 = U[:, 0]  # [d]
        acc = []
        for _ in range(n_batches):
            try:
                x, _, _ = next(it)
            except StopIteration:
                break
            x = x.to(device)
            groups = get_params_for_layers_fn()
            deltas = [None] * len(groups)
            deltas[ℓ] = eps * u0
            with apply_flat_deltas_temporarily(groups, deltas, reconstruct_from_flat_fn):
                p1 = model(x.float())
            p0 = model(x.float())
            diff = (p1.view(-1) - p0.view(-1)).norm() / (eps + 1e-8)
            acc.append(diff.item())
        if acc:
            sens[ℓ] = float(np.mean(acc))
    return sens

@torch.no_grad()
def robustness_curve_auc_lite(
    model,
    dataloader,
    basis: Dict[int, Dict[str, torch.Tensor]],
    get_params_for_layers_fn,
    reconstruct_from_flat_fn,
    # ↓ smaller defaults for CPU
    scales: List[float] = (0.0, 0.4, 0.8),
    trials_per_scale: int = 1,
    max_batches: int = 1,               # evaluate on just N small batches
    use_top1_only: bool = True,         # use top-1 PC direction per layer
    zero_out_last_layer: bool = True,
    out_idx: Optional[int] = None,
) -> Tuple[Dict[float, float], float]:
    """
    CPU-friendly robustness: evaluates only on up to `max_batches` cached batches,
    with few scales and one trial per scale by default. AUC by trapezoid rule.
    """
    device = next(model.parameters()).device
    model.eval()

    # 1) cache a few batches (prevents re-iterating the whole loader per scale)
    cached = []
    it = iter(dataloader)
    for _ in range(max_batches):
        try:
            x, y, _ = next(it)
        except StopIteration:
            break
        cached.append((x.to(device, non_blocking=False), y.to(device, non_blocking=False)))
    if not cached:
        mae_by_scale = {float(s): 0.0 for s in scales}
        return mae_by_scale, 0.0

    # 2) prepare one set of deltas per scale (don’t resample per trial on CPU)
    groups = get_params_for_layers_fn()
    L = len(groups)
    deltas_per_scale: List[Tuple[float, List[Optional[torch.Tensor]]]] = []

    for s in scales:
        deltas = [None] * L
        for ℓ, entry in basis.items():
            if zero_out_last_layer and out_idx is not None and ℓ == out_idx:
                continue
            U = entry["U"].to(device)
            if U.numel() == 0:
                continue
            r = U.shape[1]
            if use_top1_only:
                z = torch.zeros(r, device=device); z[0] = 1.0
            else:
                z = torch.randn(r, device=device)
                z = z / (z.norm() + 1e-8)
            deltas[ℓ] = U @ (float(s) * z)
        deltas_per_scale.append((float(s), deltas))

    # 3) evaluate (few batches, few scales)
    mae_by_scale: Dict[float, float] = {}
    for s, deltas in deltas_per_scale:
        # Optional “trials” loop kept for API parity, but on CPU we just reuse deltas
        trial_mae = []
        for _ in range(trials_per_scale):
            with apply_flat_deltas_temporarily(groups, deltas, reconstruct_from_flat_fn):
                losses = []
                for x, y in cached:
                    p = model(x.float())
                    losses.append(_mae(p, y).item())
            if losses:
                trial_mae.append(float(np.mean(losses)))
        mae_by_scale[s] = float(np.mean(trial_mae)) if trial_mae else 0.0

    # 4) trapezoid AUC
    xs = np.asarray(sorted(mae_by_scale.keys()), dtype=np.float64)
    ys = np.asarray([mae_by_scale[float(x)] for x in xs], dtype=np.float64)
    auc = float(np.trapz(ys, xs))
    return mae_by_scale, auc


@torch.no_grad()
def robust_mae_under_E(
    model,
    dataloader,
    basis: Dict[int, Dict[str, torch.Tensor]],
    get_params_for_layers_fn,
    reconstruct_from_flat_fn,
    m: int = 5,
    step_scale: Optional[float] = None,   # if None use per-layer basis["scale"]
    zero_out_last_layer: bool = True,
    out_idx: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Returns: (MAE_clean, MAE_E_mean, MAE_E_worst)
    For each trial i in 1..m sample delta_ℓ = scale * U_ℓ z / ||z||, apply to all layers, eval MAE.
    """
    device = next(model.parameters()).device
    model.eval()

    # Clean pass
    clean_losses = []
    for x, y, _ in dataloader:
        x = x.to(device); y = y.to(device)
        p = model(x.float())
        clean_losses.append(_mae(p, y).item())
    mae_clean = float(np.mean(clean_losses)) if clean_losses else 0.0

    # Robust passes
    maes = []
    layer_count = len(get_params_for_layers_fn())
    for _ in range(m):
        groups = get_params_for_layers_fn()
        deltas = [None] * layer_count
        # sample one delta per layer
        for ℓ, entry in basis.items():
            if zero_out_last_layer and out_idx is not None and ℓ == out_idx:
                continue
            U = entry["U"].to(device)
            if U.numel() == 0:
                continue
            r = U.shape[1]
            z = torch.randn(r, device=device)
            z = z / (z.norm() + 1e-8)
            scale = float(entry.get("scale", 0.4)) if step_scale is None else float(step_scale)
            deltas[ℓ] = U @ (scale * z)  # [d]
        # eval with temporary deltas
        trial_losses = []
        with apply_flat_deltas_temporarily(groups, deltas, reconstruct_from_flat_fn):
            for x, y, _ in dataloader:
                x = x.to(device); y = y.to(device)
                p = model(x.float())
                trial_losses.append(_mae(p, y).item())
        maes.append(float(np.mean(trial_losses)) if trial_losses else 0.0)

    if not maes:
        return mae_clean, mae_clean, mae_clean
    return mae_clean, float(np.mean(maes)), float(np.max(maes))

@torch.no_grad()
def robustness_curve_auc(
    model,
    dataloader,
    basis: Dict[int, Dict[str, torch.Tensor]],
    get_params_for_layers_fn,
    reconstruct_from_flat_fn,
    scales: List[float] = (0.0, 0.2, 0.4, 0.6, 0.8),
    trials_per_scale: int = 3,
    zero_out_last_layer: bool = True,
    out_idx: Optional[int] = None,
) -> Tuple[Dict[float, float], float]:
    """
    Returns: (mae_by_scale, auc)
    AUC computed by trapezoidal rule over the scale grid.
    """
    device = next(model.parameters()).device
    model.eval()
    mae_by_scale: Dict[float, float] = {}

    for s in scales:
        trial_mae = []
        for _ in range(trials_per_scale):
            groups = get_params_for_layers_fn()
            deltas = [None] * len(groups)
            for ℓ, entry in basis.items():
                if zero_out_last_layer and out_idx is not None and ℓ == out_idx:
                    continue
                U = entry["U"].to(device)
                if U.numel() == 0:
                    continue
                r = U.shape[1]
                z = torch.randn(r, device=device); z = z / (z.norm() + 1e-8)
                deltas[ℓ] = U @ (float(s) * z)
            with apply_flat_deltas_temporarily(groups, deltas, reconstruct_from_flat_fn):
                losses = []
                for x, y, _ in dataloader:
                    x = x.to(device); y = y.to(device)
                    p = model(x.float()); losses.append(_mae(p, y).item())
            if losses:
                trial_mae.append(float(np.mean(losses)))
        mae_by_scale[float(s)] = float(np.mean(trial_mae)) if trial_mae else 0.0

    # trapezoid AUC
    xs = np.asarray(sorted(mae_by_scale.keys()), dtype=np.float64)
    ys = np.asarray([mae_by_scale[float(x)] for x in xs], dtype=np.float64)
    auc = float(np.trapz(ys, xs))
    return mae_by_scale, auc

@torch.no_grad()
def basis_explained_energy(
    E_layer_wise: Dict[int, torch.Tensor],
    r: int = 1
) -> Dict[int, float]:
    """
    For each layer ℓ with E[d,K], computes energy fraction captured by top-r singular values
    of centered residuals: sum_{i<=r} s_i^2 / sum_i s_i^2
    """
    frac: Dict[int, float] = {}
    for ℓ, E in E_layer_wise.items():
        X = E - E.mean(dim=1, keepdim=True)
        q = min(r, X.shape[1])
        if q < 1 or X.abs().sum() == 0:
            frac[ℓ] = 0.0
            continue
        U, S, _ = torch.svd_lowrank(X, q=q)
        # Need full norm; approximate via top-q if q small: compute total via Fro norm
        total = (X**2).sum().item()
        top   = (S[:q]**2).sum().item()
        frac[ℓ] = float(top / (total + 1e-12))
    return frac

def site_spread_stats(site_mse: Dict[int, float]) -> Tuple[float, float]:
    """
    Input: site_mse like {site_id: mse}
    Returns: (variance, p90_minus_p10) over sites
    """
    if not site_mse:
        return 0.0, 0.0
    vals = np.asarray(list(site_mse.values()), dtype=np.float64)
    var  = float(np.var(vals))
    gap  = float(np.percentile(vals, 90) - np.percentile(vals, 10))
    return var, gap



# Helper to read and reset (add in class)
def trust_clip_rate(self, reset: bool = False) -> float:
    rate = float(self._clip_hits / self._clip_total) if self._clip_total else 0.0
    if reset:
        self._clip_hits = 0; self._clip_total = 0
    return rate
