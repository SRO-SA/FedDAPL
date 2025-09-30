# eval_actor.py
import ray, torch, numpy as np, zlib, base64, json
from typing import Dict, Optional, Tuple, List
from contextlib import contextmanager

from utils import log_print
import os

# ---------- Helpers ----------
def _decode_basis_b64(b64_str: str, device, dtype) -> Dict[int, Dict[str, torch.Tensor]]:
    blob = base64.b64decode(b64_str)
    data = json.loads(zlib.decompress(blob).decode("utf-8"))
    basis = {}
    for k_str, entry in data.items():
        shape = tuple(entry["shape"])
        dt = np.float16 if entry["dtype"] == "float16" else np.float32
        raw = base64.b64decode(entry["u_b64"])
        U_np = np.frombuffer(raw, dtype=dt).reshape(shape).astype(np.float32)
        U = torch.from_numpy(U_np).to(device=device, dtype=dtype)
        basis[int(k_str)] = {"U": U, "scale": float(entry["scale"])}
    return basis

def _mae(pred, target):
    pred = pred.view_as(target)
    return torch.nn.functional.l1_loss(pred, target, reduction="mean")

def _get_params_for_layers(model):
    return model.get_params_for_layers()

@torch.no_grad()
def _split_like(flat: torch.Tensor, ref_params: List[torch.nn.Parameter]):
    sizes = [p.numel() for p in ref_params]
    parts = list(torch.split(flat, sizes))
    return [part.view_as(p) for part, p in zip(parts, ref_params)]

@contextmanager
def _apply_flat_deltas_temporarily(model, deltas_by_layer: Dict[int, torch.Tensor]):
    groups = model.get_params_for_layers()
    saved = [[p.detach().clone() for p in grp] for grp in groups]
    try:
        for ℓ, delta_flat in deltas_by_layer.items():
            if delta_flat is None:
                continue
            ref = groups[ℓ]
            for p, d in zip(ref, _split_like(delta_flat.to(p.device, p.dtype), ref)):
                p.data.add_(d)
        yield
    finally:
        for grp, sgrp in zip(groups, saved):
            for p, ps in zip(grp, sgrp):
                p.data.copy_(ps)

def _project_first_basis_component(g: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    # Fast 1D projection onto u1
    u1 = U[:, 0] if U.dim() == 2 else U.view(-1)
    denom = (u1.norm() ** 2 + 1e-12)
    return (g @ u1) * u1 / denom

def quick_robustness_proxy_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    basis: Dict[int, Dict[str, torch.Tensor]],
    out_idx: int,
    zero_out_last_layer: bool = True,
) -> float:
    """
    One batch, one backward: estimate mean ||P_E(g)||/||g|| across layers.
    Requires gradients; do NOT wrap caller in @torch.no_grad().
    """
    device = next(model.parameters()).device
    model.train()  # need grads

    # clear old grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_(); p.grad.zero_()

    pred = model(x.float())
    loss = torch.nn.functional.mse_loss(pred.view_as(y), y)
    loss.backward()

    groups = _get_params_for_layers(model)
    ratios = []
    for ℓ, plist in enumerate(groups):
        if zero_out_last_layer and ℓ == out_idx:
            continue
        if ℓ not in basis:
            continue
        U = basis[ℓ]["U"].to(device)
        if U.numel() == 0:
            continue
        g_list = [(p.grad if p.grad is not None else torch.zeros_like(p)).view(-1) for p in plist]
        g = torch.cat(g_list)
        n = g.norm().item()
        if n == 0.0:
            continue
        gE = _project_first_basis_component(g, U)
        ratios.append(gE.norm().item() / (n + 1e-8))
    return float(np.mean(ratios)) if ratios else 0.0


@ray.remote(num_cpus=1, num_gpus=0.3, max_restarts=1, max_task_retries=1)
class EvalActor:
    def __init__(self, client_cfg: dict = None, use_cuda: bool = True):
        self.client_cfg = client_cfg or {}
        self.use_cuda = bool(use_cuda)
        gpu_ids = ray.get_gpu_ids()  # e.g., [1.0] or [0.0]; empty if none
        # ---- Record assigned GPUs but DO NOT touch CUDA yet ----

        if not gpu_ids:
            msg = "ERROR: No GPU assigned to EvalActor by Ray (ray.get_gpu_ids() returned empty). Aborting."
            try:
                log_print(msg)
            finally:
                raise RuntimeError(msg)

        # Make device mapping explicit (Ray already masks, but this helps debugging)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(i)) for i in gpu_ids)

        # lazy-init fields
        self._cuda_ready = False
        self.device = torch.device("cuda:0")  # will be validated on first use
        # log_print(f"EvalActor constructed. Ray GPU IDs {gpu_ids}. (CUDA init deferred)", context="ACTOR")
 
 
    def _ensure_cuda_ready(self):
        if self._cuda_ready:
            return
        if not self.use_cuda:
            raise RuntimeError("EvalActor requires CUDA (use_cuda=False).")
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}).")
        if torch.cuda.device_count() == 0:
            raise RuntimeError(f"No visible CUDA devices (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}).")
        torch.cuda.set_device(0)  # assigned GPU is cuda:0 inside this masked process
        try:
            torch.set_default_device(self.device)
        except Exception:
            pass
        _cudnn = getattr(torch.backends, "cudnn", None)
        if _cudnn is not None:
            try:
                _cudnn.benchmark = True
            except Exception:
                pass
        self._cuda_ready = True
        log_print("CUDA ready inside actor.", context="ACTOR")
 

    def ping(self):
        return 1

    def _state_dict_to_device_dtype(self, state_dict: Dict[str, np.ndarray], ref: Dict[str, torch.Tensor]):
        """Convert numpy -> torch with correct dtype and device of model."""
        out = {}
        for k, v in state_dict.items():
            tgt_dtype = ref[k].dtype
            out[k] = torch.as_tensor(v, dtype=tgt_dtype, device=self.device)
        return out

    def _assert_model_on_cuda(self, model: torch.nn.Module):
        if not next(model.parameters()).is_cuda:
            msg = "ERROR: Model parameters are on CPU; expected CUDA. Aborting."
            try:
                log_print(msg)
            finally:
                raise RuntimeError(msg)

    def run(
        self,
        state_dict: Dict[str, np.ndarray],     # CPU numpy weights
        e_basis_b64: Optional[str],            # packed basis string (or None)
        x_np: np.ndarray, y_np: np.ndarray,    # one small batch
        out_idx: int,
        zero_out_last_layer: bool = True,
    ) -> Dict[str, float]:
        from model import BrainCancer
        self._ensure_cuda_ready()
        # 1) Rebuild model and load weights on self.device
        model = BrainCancer().to(self.device)
        self._assert_model_on_cuda(model)
        log_print("Model built and moved to device.", context="ACTOR")
        ref_sd = model.state_dict()
        sd_dev = self._state_dict_to_device_dtype(state_dict, ref_sd)
        model.load_state_dict(sd_dev, strict=True)
        model.eval()

        # 2) Move batch to device and ensure shapes match
        x = torch.from_numpy(x_np).to(self.device, non_blocking=True)
        y = torch.from_numpy(y_np).to(self.device, non_blocking=True)

        # Forward (clean MAE) under no_grad (this part doesn't need grads)
        with torch.no_grad():
            mae0 = _mae(model(x.float()), y).item()

        # 3) Decode basis if provided
        basis = {}
        log_print("start decoding basis", context="ACTOR")
        if e_basis_b64:
            basis = _decode_basis_b64(e_basis_b64, device=self.device, dtype=next(model.parameters()).dtype)

        # 4) Robustness proxy (needs grads)
        robust = 0.0
        if basis:
            robust = quick_robustness_proxy_batch(model, x, y, basis, out_idx, zero_out_last_layer)
        log_print(f"EvalActor done: MAE={mae0}, robustness proxy={robust}", context="ACTOR")
        return {"val_mae_batch": float(mae0), "robustness_proxy": float(robust)}

    @torch.no_grad()
    def auc_on_cached_batches(
        self,
        state_dict: Dict[str, np.ndarray],                    # numpy tensors ok
        e_basis_b64: Optional[str],                           # packed basis string
        batches: List[Tuple[np.ndarray, np.ndarray]],         # small list of (x_np, y_np)
        out_idx: int,
        zero_out_last_layer: bool = True,
        scales: Tuple[float, ...] = (0.0, 0.4, 0.8),
        trials_per_scale: int = 1,
    ) -> Dict[str, float]:
        from model import BrainCancer
        self._ensure_cuda_ready()
        model = BrainCancer().to(self.device)
        self._assert_model_on_cuda(model)
        model.eval()
        log_print(f"device is {self.device}")

        log_print("start decoding basis")
        basis = {}
        if e_basis_b64:
            basis = _decode_basis_b64(e_basis_b64, device=self.device, dtype=next(model.parameters()).dtype)

        cached = [(torch.from_numpy(x).to(self.device, non_blocking=True),
                   torch.from_numpy(y).to(self.device, non_blocking=True)) for x, y in batches]

        mae_by_scale = {}
        log_print("start AUC eval")
        for s in scales:
            trial_mae = []
            for _ in range(trials_per_scale):
                deltas = {}
                for ℓ, entry in basis.items():
                    if zero_out_last_layer and ℓ == out_idx:
                        continue
                    U = entry["U"]
                    if U.numel() == 0:
                        continue
                    r = U.shape[1]
                    # U is already on the right device; sample z there too
                    z = torch.randn(r, device=U.device)
                    z = z / (z.norm() + 1e-8)
                    deltas[ℓ] = (float(s) * (U @ z))

                with _apply_flat_deltas_temporarily(model, deltas):
                    losses = []
                    for x, y in cached:
                        p = model(x.float())
                        losses.append(_mae(p, y).item())
                if losses:
                    trial_mae.append(float(np.mean(losses)))
            mae_by_scale[float(s)] = float(np.mean(trial_mae)) if trial_mae else 0.0

        xs = np.asarray(sorted(mae_by_scale.keys()), dtype=np.float64)
        ys = np.asarray([mae_by_scale[float(x)] for x in xs], dtype=np.float64)
        auc = float(np.trapz(ys, xs))
        return {"mae_by_scale": {float(k): float(v) for k, v in mae_by_scale.items()}, "auc": auc}
