# 
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Utility Layers (RMSNorm, SwiGLU, Token MLP, Self-Attention)
# ------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path: fused functional in PyTorch 2.9
        if hasattr(F, "rms_norm"):
            # normalize over last dim; use your existing learned scale
            return F.rms_norm(x, self.weight.shape, weight=self.weight, eps=self.eps)

        # Fallback: AMP-safe stats in float32, then cast back
        orig_dtype = x.dtype
        x32 = x if x.dtype == torch.float32 else x.float()
        inv_rms = torch.rsqrt(x32.mul(x32).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = (x32 * inv_rms).to(orig_dtype)
        return x_norm * self.weight.to(orig_dtype)




class SwiGLU(nn.Module):
    """SwiGLU feedforward gating (no bias)"""
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        # project to 2*d_hidden for gate, return to d_model
        self.w_in = nn.Linear(d_model, 2 * d_hidden, bias=False)
        self.w_out = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(u) * v)


class ChannelMLP(nn.Module):
    """standard simple pre-norm -> SwiGLU -> residual"""
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        d_hidden = int(mlp_ratio * d_model)
        self.norm = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.ff(h)
        h = self.drop(h)
        return x + h


class TokenMLPMixer(nn.Module):
    """
    token-mixing MLP (Mixer-style) applied along sequence length L suitable when L <= D (papers Sudoku case) and you want attention-free mixing
    """
    def __init__(self, seq_len: int, mlp_ratio_tokens: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(mlp_ratio_tokens * seq_len)
        self.lin1 = nn.Linear(seq_len, hidden, bias=False)
        self.lin2 = nn.Linear(hidden, seq_len, bias=False)
        self.act = nn.GELU()  # simple activation for token mixing
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> mix along L independently per channel
        b, l, d = x.shape
        h = x.transpose(1, 2)           # [B, D, L]
        h = self.lin2(self.drop(self.act(self.lin1(h))))
        h = h.transpose(1, 2)           # [B, L, D]
        return h


class SelfAttention(nn.Module):
    """Multi-head self-attention (no bias) with pre-norm and residual"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L, D]
        b, l, d = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to heads
        def reshape_heads(t):
            return t.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, Hd]

        q, k, v = map(reshape_heads, (q, k, v))
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L]
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        attn = attn_scores.softmax(dim=-1)
        attn = self.drop(attn)
        context = torch.matmul(attn, v)               # [B, H, L, Hd]
        context = context.transpose(1, 2).contiguous().view(b, l, d)  # [B, L, D]
        out = self.drop(self.proj(context))
        return x + out


class TRMBlock(nn.Module):
    """
    One block = token mixing (attention or token-MLP) + channel MLP, both with pre-norm and residual
    """
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        use_attention: bool = True,
        n_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        token_mlp_ratio: float = 2.0,
    ):
        super().__init__()
        if use_attention:
            self.mixer = SelfAttention(d_model, n_heads=n_heads, dropout=dropout)
        else:
            self.pre = RMSNorm(d_model)
            self.token_mlp = TokenMLPMixer(seq_len, mlp_ratio_tokens=token_mlp_ratio, dropout=dropout)
            self.drop = nn.Dropout(dropout)

        self.use_attention = use_attention
        self.channel_mlp = ChannelMLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            x = self.mixer(x)
        else:
            # pre-norm + token MLP + residual
            h = self.pre(x)
            h = self.token_mlp(h)
            h = self.drop(h)
            x = x + h
        # channel MLP (pre-norm inside)
        x = self.channel_mlp(x)
        return x


class TinySharedNet(nn.Module):
    """
    shared tiny network used for BOTH:
      - z update: net(x + y + z)
      - y update: net(y + z)
    (Two layers by default; Section 4.4 “Less is more”.)
    """
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        n_layers: int = 2,
        use_attention: bool = True,
        n_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        token_mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TRMBlock(
                d_model=d_model,
                seq_len=seq_len,
                use_attention=use_attention,
                n_heads=n_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                token_mlp_ratio=token_mlp_ratio,
            ) for _ in range(n_layers)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            h = blk(h)
        return h


# ------------------------------
# Model configuration and heads
# ------------------------------

@dataclass
class TRMConfig:
    # embeddings and shapes
    input_vocab_size: int
    output_vocab_size: int
    seq_len: int
    d_model: int = 512

    # Tiny shared network
    n_layers: int = 2
    use_attention: bool = True
    n_heads: int = 8
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    token_mlp_ratio: float = 2.0

    # recursion hyperparameters
    n: int = 6           # inner latent recursion loops (per §4.1 & Figure 1)
    T: int = 3           # number of full recursions per supervision step (T-1 no_grad, 1 with grad)

    # gradient truncation (optional): number of LAST "net calls" in the final recursion to keep grads
    # one full recursion process performs (n + 1) calls: n times z=net(x+y+z), then once y=net(y+z)
    # if None, backpropagate through all (n + 1) calls
    k_last_ops: Optional[int] = None
    stabilize_input_sums: bool = True
    
    use_pointer_output: bool = False  # set True for TSP
    


class OutputHead(nn.Module):
    """maps hidden y to token logits; used as reverse embedding / decoder"""
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B, L, D] -> logits: [B, L, V]
        return self.proj(self.norm(y))


class HaltHead(nn.Module):
    """
    Halting head: predicts p(halt) from y, pool over sequence and output a single logit
    Simplified ACT (Sec. 4.6): a single halting probability trained with BCE on (yy_pred==y_true),
    with no additional continue head and no second forward pass
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.fc = nn.Linear(d_model, 1, bias=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B, L, D] -> logits: [B]
        h = self.norm(y).mean(dim=1)   # simple mean pool over tokens
        logit = self.fc(h).squeeze(-1)
        return logit


# ------------------------------
# TRM Model
# ----------------------------

class TRM(nn.Module):
    """
    Tiny Recursion Model (TRM)
    y ≡ z_H (current predicted solution), z ≡ z_L (latent reasoning)

    Implements:
      - embedded input x: [B, L] -> [B, L, D]
      - latent (y, z) both in hidden space [B, L, D]
      - full recursion process:
            do n updates of z via z = net(x + y + z)        # the recursive reasoning
            then 1 update of y via y = net(y + z)           # the answer refinement
      - deep recursion with T loops: T-1 no_grad passes, then 1 pass with gradients
      - output head produces logits over output vocabulary
      - halting head produces a per-example logit (p_halt) using only the final y

    notes:
      * set use_attention=False to use an attention-free token MLP mixer (Section 4.5)
      * set k_last_ops to backprop only through the last k net-calls in the final recursion
        (a generalization of the paper k last steps experiment). if None, backprop through all
    """
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        self.cfg = cfg

        # input embedding and output head (reverse embedding)
        self.input_emb = nn.Embedding(cfg.input_vocab_size, cfg.d_model)
        self.shared_net = TinySharedNet(
            d_model=cfg.d_model,
            seq_len=cfg.seq_len,
            n_layers=cfg.n_layers,
            use_attention=cfg.use_attention,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            mlp_ratio=cfg.mlp_ratio,
            token_mlp_ratio=cfg.token_mlp_ratio,
        )
        self.halt_head = HaltHead(cfg.d_model)

        if cfg.use_pointer_output:
            self.pointer_head = PointerOutputHead(cfg.d_model)  # defined elsewhere
            self.output_head = None
        else:
            self.output_head = OutputHead(cfg.d_model, cfg.output_vocab_size)
            self.pointer_head = None
            

    # state helpers

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        initialize (y, z) states as zeros in hidden space
        shapes: [B, L, D]
        """
        if device is None:
            device = next(self.parameters()).device
        y0 = torch.zeros(batch_size, self.cfg.seq_len, self.cfg.d_model, device=device)
        z0 = torch.zeros_like(y0)
        return y0, z0

    @staticmethod
    def _sum_inputs(tensors: List[torch.Tensor], stabilize: bool = False) -> torch.Tensor:
        h = sum(tensors)
        if stabilize:
            h = h / math.sqrt(len(tensors))
        return h

    # core recursion

    def _net(self, h: torch.Tensor) -> torch.Tensor:
        """pass through the tiny shared network"""
        return self.shared_net(h)



    def latent_recursion(
        self,
        x_h: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n: Optional[int] = None,
        k_last_ops: Optional[int] = None,
        track_grads: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.cfg.n if n is None else n

        total_ops = n + 1  # n z-updates + 1 y-update
        if (k_last_ops is None) or (k_last_ops >= total_ops):
            cutoff = 0
        else:
            cutoff = total_ops - int(k_last_ops)

        op_index = 0

        # n times: z = net(x + y + z)
        for _ in range(n):
            h_z = self._sum_inputs([x_h, y, z], stabilize=self.cfg.stabilize_input_sums)
            ctx = nullcontext() if (track_grads and op_index >= cutoff) else torch.no_grad()
            with ctx:
                z = self._net(h_z)
            op_index += 1

        # once: y = net(y + z)
        h_y = self._sum_inputs([y, z], stabilize=self.cfg.stabilize_input_sums)
        ctx = nullcontext() if (track_grads and op_index >= cutoff) else torch.no_grad()
        with ctx:
            y = self._net(h_y)
        op_index += 1
        
        return y, z



    def deep_recursion(
        self,
        x_h: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n: Optional[int] = None,
        T: Optional[int] = None,
        k_last_ops: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        perform deep recursion (T loops)
          - T-1 loops with no gradients (improve the state cheaply)
          - Final loop with gradients (backprop through the full recursion process of n z-updates + 1 y-update; optionally truncated via k_last_ops)
        returns:
          - y_next, z_next: detached states to be used as initialization for the next supervision step
          - logits: token logits; pointer head if use_pointer_output else reverse embedding
          - halt_logit: scalar logit per example for halting (single-pass ACT)
        """
        n = self.cfg.n if n is None else n
        T = self.cfg.T if T is None else T

        # T-1 improvement loops without gradients
        for _ in range(max(0, T - 1)):
            with torch.no_grad():
                y, z = self.latent_recursion(x_h, y, z, n=n, k_last_ops=None, track_grads=False)

        # Final loop with gradients; optionally backprop only through the last k ops
        y, z = self.latent_recursion(x_h, y, z, n=n, k_last_ops=k_last_ops, track_grads=True)
        
        # Heads
        if self.cfg.use_pointer_output:
            # candidates can be x_h (simple and effective for TSP)
            logits = self.pointer_head(y, x_h, y_state=y)     # [B, L, L]
        else:
            logits = self.output_head(y)           # [B, L, V]
            
        halt_logit = self.halt_head(y)             # [B]

        # detach states before returning (deep supervision step carries detached states)
        return y.detach(), z.detach(), logits, halt_logit

    #public APIs

    def embed_input(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed tokenized input x: [B, L] -> [B, L, D]
        already have x in hidden space, bypass and feed it directly to deep_recursion)
        """
        return self.input_emb(x_tokens)

    def forward_step(
        self,
        x_tokens: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        *,
        n: Optional[int] = None,
        T: Optional[int] = None,
        k_last_ops: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One supervision step (described in 4.1 and illustrated in Figure 1 of the paper):
          1 x = embed(x_tokens)
          2 run T deep-recursions (T-1 no_grad, 1 with grad)
          3 return next (y, z) to carry across steps, plus logits and halt_logit
        """
        x_h = self.embed_input(x_tokens)
        if (y is None) or (z is None):
            y, z = self.init_state(batch_size=x_tokens.size(0), device=x_tokens.device)
        return self.deep_recursion(x_h, y, z, n=n, T=T, k_last_ops=k_last_ops)




#####################################################
##### DIFFERENT APPROACH NEEDED FOR PROBLEMS LIKE TSP

# in trm.py
# PointerOutputHead
class PointerOutputHead(nn.Module):
    def __init__(self, d_model: int, bilinear: bool = True):
        super().__init__()
        self.norm_q = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)
        self.bilinear = bilinear
        if bilinear:
            self.W = nn.Parameter(torch.empty(d_model, d_model))
            nn.init.xavier_uniform_(self.W)
        # learnable mix of candidates
        self.beta_y = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        y: torch.Tensor,          # [B, L, D] queries
        x_h: torch.Tensor,        # [B, L, D] geometric content
        y_state: torch.Tensor | None = None  # [B, L, D] optional
    ) -> torch.Tensor:
        q = self.norm_q(y)
        k_in = x_h if y_state is None else (x_h + self.beta_y * y_state)
        k = self.norm_k(k_in)
        if self.bilinear:
            qW = torch.einsum("bld,dk->blk", q, self.W)
            logits = torch.einsum("blk,bmk->blm", qW, k)  # [B, L, L]
        else:
            logits = torch.einsum("bld,bmd->blm", q, k)    # [B, L, L]
        return logits / math.sqrt(q.size(-1))
