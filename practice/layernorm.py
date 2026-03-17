import torch
import torch.nn as nn
import torch.functional as F


class BatchNorm(nn.Module):
    def __init__(self, momentum: float=0.1, eps: float=1e-5) -> None:
        super().__init__()
        
##################################################

# [*] Caution:p
#   1. first add eps then sqrt

class LayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float=1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(hidden_dim)) # 必须传入tensor，且gamma初始化为1, beta为0
        self.beta  = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            - x: (b, n, d)
        Returns:
            - x: (b, n, d)
        """
        mean = torch.mean(x, dim=-1, keepdim=True) # (b, d, 1)
        var  = torch.var(x, dim=-1, keepdim=True) # (b, d, 1)
        z = (x - mean) / torch.sqrt(var + self.eps)
        
        x = z * self.gamma + self.beta
        return x 
        
x = torch.randn(2, 4) # (n, d)
ln = LayerNorm(4)
x_rescale = ln(x)
print(f"x: {x}\nLN of x: {x_rescale}")

##################################################

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float=1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        
    def forward(self, x: torch.Tensor):
        return x * torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        ) * self.weight

x = torch.randn(2, 4) # (n, d)
rmsnorm = RMSNorm(4)
x_rescale = rmsnorm(x)
print(f"x: {x}\nRMSNorm of x: {x_rescale}")

##################################################

# [*] Caution:
#   1. cond_dim
#   2. only the last layer needs to be zero initialized
#   3. unqueeze
#   4. 1 + gamma

class AdaLN(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.mlp = MLP([cond_dim, hidden_dim*6, hidden_dim*3])
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # Ada-zero initialization
        output = self.mlp.layers[-1]
        nn.init.constant_(output.weight, 0)
        nn.init.constant_(output.bias, 0)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Args:
            x: (b, n, d) hidden state
            c: (b, cond_dim) condition
        Returns:
            x: (b, n, d) hidden state
            a: (b, 1, d) alpha, gating factor
        """
        g, b, a = self.mlp(c).chunk(3, dim=-1) # (b, d)
        x_norm = self.ln(x)
        x = x_norm*(1 + g.unsqueeze(1)) + b.unsqueeze(1)
        return x, a.unsqueeze(1)

##################################################

class MLP(nn.Module):
    def __init__(self, dims: list | tuple, activation=nn.SiLU):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i+1])
            self.layers.append(layer)
            # Add activation, except for output layer
            if i < len(dims) - 2:
                self.layers.append(activation())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    