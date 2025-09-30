# test_ray_actor_min.py
import os, time
import ray

def p(msg): print(msg, flush=True)

# ----------- Good: no CUDA work in __init__ -----------
@ray.remote(num_cpus=1, num_gpus=1)
class CudaActor:
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda
        self._cuda_ready = False
        self.device = None  # set later
        # keep __init__ instant
    def ping(self):
        return "pong"
    def init_cuda(self):
        import torch
        if not self.use_cuda:
            return {"ok": False, "msg": "use_cuda=False"}
        if not torch.cuda.is_available():
            return {"ok": False, "msg": "torch.cuda.is_available() == False",
                    "env": os.environ.get("CUDA_VISIBLE_DEVICES")}
        if torch.cuda.device_count() == 0:
            return {"ok": False, "msg": "torch.cuda.device_count() == 0",
                    "env": os.environ.get("CUDA_VISIBLE_DEVICES")}
        torch.cuda.set_device(0)             # Ray masks; 0 is our assigned GPU
        self.device = "cuda:0"
        try:
            torch.set_default_device(self.device)
        except Exception:
            pass
        # do a tiny CUDA op
        x = torch.randn(4, 4, device=self.device)
        self._cuda_ready = True
        return {"ok": True, "device": str(x.device), "sum": float(x.sum().item())}
    def compute(self, n: int = 1_000_00):
        import torch
        if not self._cuda_ready:
            r = self.init_cuda()
            if not r.get("ok"):
                return {"ok": False, "msg": f"init_cuda failed: {r}"}
        x = torch.randn(n, device=self.device)
        return {"ok": True, "mean": float(x.mean().item())}

# ----------- Bad: touches CUDA inside __init__ (can hang) -----------
@ray.remote(num_cpus=1, num_gpus=1)
class BadCudaActor:
    def __init__(self):
        # WARNING: This can hang if GPU isn’t ready/schedulable
        import torch
        torch.cuda.set_device(0)
        _ = torch.randn(1, device="cuda")
    def ping(self):
        return "pong"

if __name__ == "__main__":
    # Init Ray (adjust num_gpus if needed)
    ray.init(num_cpus=2, num_gpus=1, include_dashboard=False, ignore_reinit_error=True)
    p(f"Cluster: {ray.cluster_resources()}")
    p(f"Avail  : {ray.available_resources()}")

    # ---- Test good actor (should be instant) ----
    p("\nCreating CudaActor...")
    a = CudaActor.options(max_restarts=1, max_task_retries=1).remote(True)
    p("Pinging CudaActor...")
    pong = ray.get(a.ping.remote(), timeout=60)   # should NOT time out
    p(f"Ping -> {pong}")

    p("Init CUDA inside actor...")
    init_info = ray.get(a.init_cuda.remote(), timeout=120)
    p(f"init_cuda -> {init_info}")

    p("Compute on GPU...")
    res = ray.get(a.compute.remote(200_000), timeout=120)
    p(f"compute -> {res}")

    # ---- Optional: demonstrate how a bad __init__ can block ----
    # Uncomment to see if your environment hangs here.
    # p("\nCreating BadCudaActor (may hang if CUDA in __init__ is problematic)...")
    # b = BadCudaActor.options(max_restarts=0, max_task_retries=0).remote()
    # p("Pinging BadCudaActor...")
    # pong2 = ray.get(b.ping.remote(), timeout=60)
    # p(f"Ping -> {pong2}")

    p("\nDone.")
    ray.shutdown()
