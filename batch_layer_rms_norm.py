import torch
import torch.nn as nn
import torch.functional as F

class CustomLayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super(CustomLayerNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(self.hidden_dim))
        self.beta = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdims = True)
        var = x.var(dim = -1, keepdims = True)
        x_norm = (x - mean)/torch.sqrt(var + self.eps)
        return self.alpha * x_norm + self.beta
    
x = torch.randn(2, 4)
custom_layer_norm = CustomLayerNorm(4)
normalized_x = custom_layer_norm(x)
print(normalized_x)

##############################################

class CustomRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x) 
        return x * torch.rsqrt(
            x.pow(2).mean(dim = -1, keepdim = True) 
            + self.eps
            )
    
    def forward(self, x: torch.Tensor):
        # x.shape = [B, seq_len, dim] 
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)
    
x = torch.randn(2, 4)
custom_rms_norm = CustomRMSNorm(4)
normalized_x = custom_rms_norm(x)
print(normalized_x)

##############################################

class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta  = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, D) 或 (B, D)
        """
        # 如果是 (B, N, D)，BN 通常在 D 维度上归一化，合并 B 和 N
        if x.dim() == 3:
            B, N, D = x.shape
            x_reshaped = x.reshape(-1, D)
        else:
            x_reshaped = x

        if self.training:
            # 计算当前 Batch 的均值和方差 (在 0 维度上，即所有样本)
            batch_mean = x_reshaped.mean(dim=0)
            batch_var = x_reshaped.var(dim=0, unbiased=False)
            
            # 更新全局统计量 (Exponential Moving Average)
            # running = (1-m) * running + m * batch
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            curr_mean = batch_mean
            curr_var = batch_var
        else:
            # 推理阶段：直接使用 Buffer 中的全局统计量
            curr_mean = self.running_mean
            curr_var = self.running_var

        # 归一化
        x_norm = (x_reshaped - curr_mean) / torch.sqrt(curr_var + self.eps)
        
        # 仿射变换
        out = x_norm * self.gamma + self.beta
        
        # 如果输入是 3D，还原形状
        if x.dim() == 3:
            out = out.reshape(B, N, D)
            
        return out