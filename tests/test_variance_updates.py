"""Sanity check: verify variance head evolves over 50 and 100 steps."""
import math
import torch
import torch.nn as nn
from models.model_mode import Uncertainty

def test_variance_updates():
    device = "cpu"
    n_steps = 100
    checkpoints = [0, 9, 49, 99]

    for var_type in ("linear", "mlp", "separate_agg"):
        for var_init in (False, True):
            opt = {
                "backbone": "ResNet18",
                "descriptors_dimension": 512,
                "model_mode": "uncertainty",
                "var_head_type": var_type,
                "var_init": var_init,
                "variance_activation": "softplus",
            }
            model = Uncertainty(opt).to(device)
            model.freeze_base()

            trainable = [p for p in model.parameters() if p.requires_grad]
            tag = f"{var_type:>14s}, init={var_init!s:<5s}"

            optimizer = torch.optim.Adam(trainable, lr=1e-3)
            criterion = nn.GaussianNLLLoss()

            snapshots = {}
            for step in range(n_steps):
                x = torch.randn(4, 3, 224, 224, device=device)
                optimizer.zero_grad()
                mu, var = model(x)
                scale = math.sqrt(mu.shape[1])
                target = mu.detach() + torch.randn_like(mu) * 0.1
                loss = criterion(mu * scale, target * scale, var)
                loss.backward()
                optimizer.step()
                if step in checkpoints:
                    snapshots[step] = (var.min().item(), var.mean().item(), var.max().item(), loss.item())

            print(f"[{tag}]")
            for s in checkpoints:
                mn, mean, mx, l = snapshots[s]
                print(f"    step {s:3d}: min={mn:.4f}  mean={mean:.4f}  max={mx:.4f}  loss={l:.4f}")
            print()

    print("Done.")

if __name__ == "__main__":
    test_variance_updates()
