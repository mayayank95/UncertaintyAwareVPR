import torch
import math

class VMFLikelihoodAmos(torch.nn.Module):
    def __init__(self, d=256, eps=1e-7):
        super().__init__()
        self.d = d
        self.eps = eps

    def log_bessel_approx(self, kappa):
        v = (self.d - 1) / 2.0
        y = torch.sqrt(kappa**2 + v**2)
        integral = y - v * torch.log(v + y)
        
        log_z0 = math.log(2.0) + (self.d / 2.0) * math.log(math.pi) - math.lgamma(self.d / 2.0)
        integral_0 = v - v * math.log(2 * v)
        c = log_z0 - integral_0
        
        return integral + c

vmf = VMFLikelihoodAmos(d=512)
kappa = torch.tensor([1.0, 5.0, 10.0, 100.0, 1000.0], requires_grad=True, dtype=torch.float64)

log_Zd = vmf.log_bessel_approx(kappa)
print("Amos log_Zd:", log_Zd)

log_Zd.sum().backward()
print("Amos grad_kappa (A_d(kappa)):", kappa.grad)

# Using finite differences
kappa_eps = kappa.clone().detach() + 1e-4
log_Zd_eps = vmf.log_bessel_approx(kappa_eps)
numeric_grad = (log_Zd_eps - log_Zd.detach()) / 1e-4
print("Amos grad numerical:", numeric_grad)
