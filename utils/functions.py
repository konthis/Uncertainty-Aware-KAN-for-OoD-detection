import torch
import torch.nn as nn
import torch.nn.functional as F

import time

class ActivationFunctions(nn.Module):
    def __init__(self,gamma):
        super(ActivationFunctions,self).__init__()
        self.gamma = gamma
    def GaussianRBF(self, x):
        return torch.exp(-self.gamma*torch.square(x))

    def RBF_SiLU(self, x):
        return x*torch.exp(-self.gamma*torch.square(x))

    def RBF_Swish(self, x):
        return F.silu(x)*x*torch.exp(-self.gamma*torch.square(x))

def gradPenalty2sideCalc(x, ypred):
    gradients = torch.autograd.grad(
            outputs=ypred,
            inputs=x,
            grad_outputs=torch.ones_like(ypred),
            create_graph=True
        )[0]
    gradients = gradients.flatten(start_dim=1)
    gradPenalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradPenalty


class LogitNormLoss(nn.Module):
    def __init__(self, tau: float = 0.04, eps: float = 1e-7):
        super().__init__()
        self.tau = tau
        self.eps = eps
 
    def forward(self, logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        # Compute the L2 norm per sample
        norm = torch.norm(logits, p=2, dim=1, keepdim=True).clamp(min=self.eps)
        
        # Normalize logits
        normalized_logits = logits / (self.tau * norm)
 
        # Compute cross-entropy on normalized logits
        return F.cross_entropy(normalized_logits, targets)

class ProposedLoss(nn.Module):
    def __init__(self, lamda: float = 0.1, tau: float = 0.04, eps: float = 1e-7):
        super().__init__()
        self.tau = tau
        self.eps = eps
        self.lamda = lamda
 
    def forward(self, logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        return LogitNormLoss(self.tau,self.eps).forward(logits,targets) * (1-self.lamda) + self.lamda*F.cross_entropy(logits,targets)



def model_stats(model, input_size: tuple, device, n_warmup: int = 50, n_runs: int = 500):
    # param count flops inference etc
    model.eval()
    dummy = torch.zeros(1, *input_size).to(device)

    # Parameter count
    total_params     = sum(p.numel() for p in model.parameters())

    # Inference time — warm up then average
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        ms_per_sample = (time.perf_counter() - t0) / n_runs * 1000

    print(f"  Params (total):     {total_params:,}")
    print(f"  Inference time:     {ms_per_sample:.4f} ms/sample")
