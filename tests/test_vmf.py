import torch
from scipy.special import ive
import numpy as np

class LogBesselIv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, kappa):
        # We compute log(I_v(kappa)) using exponentially scaled Bessel function
        # I_v(kappa) = ive(v, kappa) * exp(kappa)
        # log I_v(kappa) = log(ive(v, kappa)) + kappa
        kappa_np = kappa.detach().cpu().numpy()
        ive_v = ive(v, kappa_np)
        
        # Clamp against underflow
        ive_v[ive_v <= 0] = 1e-30
        
        ctx.v = v
        ctx.save_for_backward(kappa)
        
        log_iv = torch.tensor(np.log(ive_v), dtype=kappa.dtype, device=kappa.device) + kappa
        return log_iv

    @staticmethod
    def backward(ctx, grad_output):
        v = ctx.v
        kappa, = ctx.saved_tensors
        
        # The derivative of log I_v(kappa) is I_{v+1}(kappa) / I_v(kappa) + v / kappa
        kappa_np = kappa.detach().cpu().numpy()
        ive_v = ive(v, kappa_np)
        ive_v_plus_1 = ive(v + 1, kappa_np)
        
        # Add epsilon to prevent division by zero
        ratio = ive_v_plus_1 / (ive_v + 1e-30)
        grad_kappa = torch.tensor(ratio, dtype=kappa.dtype, device=kappa.device) + (v / kappa)
        
        return None, grad_output * grad_kappa

kappa = torch.tensor([1.0, 5.0, 10.0, 100.0], requires_grad=True, dtype=torch.float64)
v = 255.5
log_iv = LogBesselIv.apply(v, kappa)
print("log_iv:", log_iv)

log_iv.sum().backward()
print("grad_kappa (exact SciPy autograd):", kappa.grad)

# Using finite differences
kappa_eps = kappa.clone().detach() + 1e-4
log_iv_eps = LogBesselIv.apply(v, kappa_eps)
numeric_grad = (log_iv_eps - log_iv.detach()) / 1e-4
print("grad numerical:", numeric_grad)

