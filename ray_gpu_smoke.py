# ray_gpu_smoke.py
import ray

def p(*a): print(*a, flush=True)

# 1) A trivial CPU actor (should always work)
@ray.remote
class CPUActor:
    def ping(self): return "pong"

# 2) A trivial GPU actor that DOES NOT import torch (just reserves a GPU)
@ray.remote(num_gpus=1)
class GPUActor:
    def ping(self): return "pong"

# 3) A GPU actor that imports torch but only touches CUDA inside a method
@ray.remote(num_gpus=1)
class TorchGPUActor:
    def ping(self): return "pong"
    def cuda_ok(self):
        import torch
        return {
            "is_available": torch.cuda.is_available(),
            "count": torch.cuda.device_count(),
        }
    def tiny_cuda(self):
        import torch
        torch.cuda.set_device(0)
        x = torch.randn(8, device="cuda")
        return float(x.sum().item())

if __name__ == "__main__":
    
    ray.shutdown()
    ray.init(num_cpus=2, num_gpus=2, include_dashboard=False, ignore_reinit_error=True)
    print("cluster:", ray.cluster_resources())
    print("avail  :", ray.available_resources())
    print("nodes  :", [{"NodeID": n.get("NodeID"), "Alive": n.get("Alive"), "Resources": n.get("Resources")} for n in ray.nodes()])

    @ray.remote(num_gpus=1)
    class P:
        def ping(self): return "pong"

    a1 = P.remote()
    a2 = P.remote()
    print("pings:", ray.get([a1.ping.remote(), a2.ping.remote()], timeout=60)) 
       
    ray.shutdown()

    ray.init(num_cpus=2, num_gpus=1, include_dashboard=False, ignore_reinit_error=True)

    p("cluster:", ray.cluster_resources())
    p("avail  :", ray.available_resources())

    # --- CPU actor
    p("\n[CPU] create...")
    a = CPUActor.remote()
    p("[CPU] ping:", ray.get(a.ping.remote(), timeout=30))

    # --- GPU actor (no torch)
    p("\n[GPU-no-torch] create...")
    g = GPUActor.remote()
    p("[GPU-no-torch] ping:", ray.get(g.ping.remote(), timeout=60))

    # --- GPU actor (torch on demand)
    p("\n[TorchGPU] create...")
    t = TorchGPUActor.remote()
    p("[TorchGPU] ping:", ray.get(t.ping.remote(), timeout=60))
    p("[TorchGPU] cuda_ok:", ray.get(t.cuda_ok.remote(), timeout=120))
    p("[TorchGPU] tiny_cuda:", ray.get(t.tiny_cuda.remote(), timeout=120))

    p("\nOK.")
    ray.shutdown()
