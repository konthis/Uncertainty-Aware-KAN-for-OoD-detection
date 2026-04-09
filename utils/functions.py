import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import elementwise_flop_counter


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
    model.eval()
    dummy = torch.zeros(1, *input_size).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # register elementwise ops to count them on FLOP
    flops = FlopCountAnalysis(model, dummy)
    flops.set_op_handle(**{
        "aten::silu":       elementwise_flop_counter(1, 0),
        "aten::softmax":    elementwise_flop_counter(1, 0),
        "aten::lt":         elementwise_flop_counter(1, 0),
        "aten::nan_to_num": elementwise_flop_counter(1, 0),
        "aten::exp":        elementwise_flop_counter(1, 0),
        "aten::pow":        elementwise_flop_counter(1, 0),
        "aten::neg":        elementwise_flop_counter(1, 0),
        "aten::mean":       elementwise_flop_counter(1, 0),
        "aten::sqrt":       elementwise_flop_counter(1, 0),
        "aten::sigmoid":       elementwise_flop_counter(1, 0),
        "aten::square":     elementwise_flop_counter(1, 0),
        "aten::mul":        elementwise_flop_counter(1, 0),
        "aten::sum":        elementwise_flop_counter(1, 0),
        "aten::log":        elementwise_flop_counter(1, 0),
        "aten::sub":        elementwise_flop_counter(1, 0),
        "aten::rsub":       elementwise_flop_counter(1, 0),
        "aten::sub_":       elementwise_flop_counter(1, 0),
        "aten::div":        elementwise_flop_counter(1, 0),
        "aten::div_":       elementwise_flop_counter(1, 0),
        "aten::add":        elementwise_flop_counter(1, 0),
        "aten::add_":       elementwise_flop_counter(1, 0),
        "aten::layer_norm": elementwise_flop_counter(1, 0),
        "prim::PythonOp.SparsemaxFunction": elementwise_flop_counter(1, 0),
        "prim::PythonOp.SparsemaxFunctionaten::":       elementwise_flop_counter(1, 0),
    })
    flops.unsupported_ops_warnings(True)
    total_flops = flops.total()

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
        ms = (time.perf_counter() - t0) / n_runs * 1000

    print(f"  Params (trainable): {trainable_params:,}")
    print(f"  FLOPs:              {total_flops:,}")
    print(f"  Inference time:     {ms:.4f} ms/sample")

    return {'params': trainable_params, 'flops': total_flops, 'ms': ms}
