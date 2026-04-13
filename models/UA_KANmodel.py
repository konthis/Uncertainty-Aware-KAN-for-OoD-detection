import torch
from torch import nn
from torch.functional import F
from typing import *

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -6.,
        grid_max: float = 6.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = nn.Parameter(torch.ones_like(torch.zeros(1))*denominator) 
        

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class UA_KANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -6.,
        grid_max: float = 6.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        denominator: float = None,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids,denominator)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def layer_uncertainty(self, x):
        spline_basis = self.rbf(x)
        distw = spline_basis / (spline_basis.sum(dim=-1, keepdim=True) + 1e-8) 
        entropy = -(distw * torch.log(distw + 1e-10)).sum(dim=-1)  
        return entropy.mean(dim=-1) 


    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )   # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y

class UA_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -6.,
        grid_max: float = 6.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        denominator: float = 1.,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            UA_KANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                denominator = denominator,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])
        ####  weights for uncertainty calculation
        self.layer_weights = [1.0 / len(self.layers)] * len(self.layers)  # uniform init

    
    def forward_with_layer_uncertainty(self, x):
        # Returns (logits, [u_layer1, u_layer2, ..., u_layerN])
        uncertainties = []
        for layer in self.layers:
            uncertainties.append(layer.layer_uncertainty(x))
            x = layer(x)
        return x, uncertainties

    def compute_layer_weights(self, train_loader, device):
        # calculate variances of each layers uncertainty score, and normalize them
        self.eval()
        collected = [[] for _ in self.layers]
        with torch.no_grad():
            for x, _ in train_loader:
                x = x.to(device)
                _, layer_us = self.forward_with_layer_uncertainty(x)
                for i, u in enumerate(layer_us):
                    collected[i].append(u.cpu())
        variances = [torch.cat(collected[i]).var().item() for i in range(len(self.layers))]
        total = sum(variances) + 1e-10
        self.layer_weights = [v / total for v in variances]
        return self.layer_weights


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forwardSoftmax(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x,dim=1)
    