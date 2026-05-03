import torch
import time
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from models.get_model import get_model

def benchmark(args, name):
    device = torch.device("cuda")
    model = get_model(args).to(device)
    model.eval()
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Measure memory (Inference on 1 image)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    # Measure latency
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Measure
    n_runs = 200
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()
    latency = (time.time() - start) * 1000 / n_runs
    
    print(f"{name}: {params:.3f}M params, {mem:.2f}MB memory, {latency:.3f}ms latency")
    return params, mem, latency

baseline_args = {
    "method": "cosplace",
    "backbone": "ResNet50",
    "descriptors_dimension": 512,
    "model_mode": "basic",
    "device": "cuda"
}

kappa_args = {
    "method": "cosplace",
    "backbone": "ResNet50",
    "descriptors_dimension": 512,
    "model_mode": "uncertainty",
    "var_head_type": "vmf_agg",
    "device": "cuda",
    "variance_activation": "softplus",
    "train_all_layers": False
}

print("Starting benchmarks...")
p1, m1, l1 = benchmark(baseline_args, "Baseline")
p2, m2, l2 = benchmark(kappa_args, "KappaPlace")

print("\n--- Summary Table ---")
print(f"Metric             | Baseline | KappaPlace | Abs Increase | Rel. Increase (%)")
print(f"Inference Latency  | {l1:.2f} ms | {l2:.2f} ms   | +{l2-l1:.2f} ms     | +{(l2-l1)/l1*100:.2f}%")
print(f"GPU Memory         | {m1:.1f} MB | {m2:.1f} MB   | +{m2-m1:.1f} MB     | +{(m2-m1)/m1*100:.2f}%")
print(f"Model Parameters   | {p1:.2f} M  | {p2:.2f} M    | +{p2-p1:.2f} M      | +{(p2-p1)/p1*100:.2f}%")
