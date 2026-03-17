from dataclasses import dataclass
# This tells type checker that either an object of the specific type is required, or None is required
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    dim: int = 4096 #Hidden Dimension across
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Required for KV Cache
    max_batch_size: int = 32
    max_seq_len: int = 2028

    device: str = None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x) 
        return x * torch.rsqrt(x.pow(2).mean(dim = -1, keepdim = True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # x.shape = [B, seq_len, dim] 
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)
    

def precompute_theta_pos_frequencies(
        head_dim: int,
        seq_len: int,
        device: str,
        theta: float = 10000.0
):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # 1 / (10000 ^ (2(i - 1)/dim))  for i = [1, 2, ... dim/2]
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)

    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()

    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freq_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freq_complex


def apply_rotary_embeddings(
        x: torch.Tensor,
        freq_complex: torch.Tensor,
        device: str
):
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1 , 2))
    freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)

    x_rotated = x_complex * freq_complex

    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int):
    if n_rep == 1: return x
    b, s, n_kv, n_hdim = x.shape
    x = x[:, :, :, None, :].expand(b, s, n_kv, n_rep, n_hdim).reshape(b, s, n_kv * n_rep, n_hdim)
    return x

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads #times KV head will be repeated.

        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freq_complex: torch.Tensor
    ):
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xq.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xq.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freq_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freq_complex, device=x.device)

        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        # (B, S, head, head_dim)
        attention = (keys.transpose(1, 2) @ values.transpose(1, 2).transpose(2, 3))/math.sqrt(self.head_dim)
        scores = torch.softmax(attention.float(), dim = -1).type_as(xq)

        output = scores @ values.transpose(1, 2) # (B, head, s, head_dim)
        output = output.transpose(1, 2).contigious().view(batch_size, seq_len, -1)

        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim /3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)


    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        return self.w3(swish * x_V)

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        out = h + self.feed_forward.forward(
            self.ffn_norm(h)
        )
        return out
    

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(0, self.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.norm, eps=args.norm_eps)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.freq_complex = precompute_theta_pos_frequencies(
                                self.dim // self.args.n_heads, 
                                self.args.max_seq_len * 2,
                                device=self.args.device
                            )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape

        h = self.token_embedding(tokens)

        freqs_complex = self.freq_complex[start_pos: start_pos + seq_len]
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        h = self.output(h).float()
        return h










