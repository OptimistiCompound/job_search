import torch
import torch.nn as nn
import torch.functional as F
import math

class TransformerBlock(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 num_block: int, 
                 num_head: int
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_block = num_block
        self.num_head = num_head
        
        modules = [TransformerLayer(hidden_dim=hidden_dim, num_head=num_head) for _ in range(num_block)]
        self.decoder = nn.Sequential(*modules)


class TransformerLayer(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 num_head: int,
                 scaling_factor: int = 4,
                ):
        super().__init__()
        
        self.ffn = MLP([hidden_dim, scaling_factor*hidden_dim, hidden_dim])
        self.attn = Attention(hidden_dim, num_head)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # prenorm
        identity = x
        x = self.ln1(x)
        x = self.attn(x)
        x = identity + x
        
        identity = x
        x = self.ln2(x)
        x = self.ffn(x)
        return x + identity


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_head: int):
        super().__init__()
        assert hidden_dim % num_head == 0
        
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        
        self.head_dim = hidden_dim // num_head
        
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        self.wo = nn.Linear(hidden_dim, hidden_dim)
        
        self.pe = nn.Parameter(hidden_dim) # Learnable PE
        
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2) # b n h*d_h -> b h n d_h
        xk = xk.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2).transpose(2, 3) # b n h*d_h -> b h d_h n
        xv = xv.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        logits = xq @ xk / math.sqrt(self.head_dim) # [*] Note self.head_dim, rather than hidden_dim
        scores = torch.softmax(logits, dim=-1).type_as(xq) # [*] Note this type_as()
        output = scores @ xv # b h n d_h
        
        # [*] Transpose back
        # alternative: 
        #   b h n d_h = output.shape()
        #   output.transpose(1, 2).reshape(b, n, h*d_h)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # b h n d_h -> b n h*d_h

        output = self.wo(output)
        
        return x


class MLP(nn.Module):
    def __init__(self, dims: list | tuple):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                self.layers.append(nn.ReLU())
    
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
        
        