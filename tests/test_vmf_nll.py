import torch
import math
import torch.nn.functional as F

class VMFLikelihoodAmos(torch.nn.Module):
    def __init__(self, d=512, eps=1e-7):
        super().__init__()
        self.d = d
        self.eps = eps

    def log_partition_function(self, kappa):
        v = (self.d - 1) / 2.0
        kappa = kappa + self.eps
        y = torch.sqrt(kappa**2 + v**2)
        integral = y - v * torch.log(v + y)
        log_z0 = math.log(2.0) + (self.d / 2.0) * math.log(math.pi) - math.lgamma(self.d / 2.0)
        integral_0 = v - v * math.log(2 * v)
        c = log_z0 - integral_0
        return integral + c

    def forward(self, mu, kappa, target):
        mu = F.normalize(mu, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        dot_prod = torch.sum(mu * target, dim=-1, keepdim=True)
        log_z_d = self.log_partition_function(kappa)
        loss = -kappa * dot_prod + log_z_d
        return loss.mean()

vmf = VMFLikelihoodAmos(d=512)
mu = torch.randn(2, 512, requires_grad=True, dtype=torch.float64)
target = mu.clone().detach() # perfect match
kappa = torch.tensor([[1.0], [10.0]], requires_grad=True, dtype=torch.float64)

loss = vmf(mu, kappa, target)
print("Loss (perfect prediction):", loss)

loss.backward()
print("Grad wrt kappa (perfect prediction - should be negative to increase kappa):", kappa.grad)

kappa.grad.zero_()
mu_bad = -target.clone().detach() # opposite prediction
loss_bad = vmf(mu_bad, kappa, target)
loss_bad.backward()
print("Grad wrt kappa (bad prediction - should be positive to decrease kappa):", kappa.grad)
