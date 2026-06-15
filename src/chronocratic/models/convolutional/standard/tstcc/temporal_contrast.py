__all__ = ["TemporalContrast"]

from typing import cast

from einops import rearrange, repeat
import torch
from torch import nn
from torch.nn import functional

# ---------------------------------------------------------------------------
# Seq_Transformer building blocks (internal to this module)
# ---------------------------------------------------------------------------


class _Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class _PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in qkv)
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        if mask is not None:
            mask = functional.pad(mask.flatten(1), (1, 0), value=True)
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float("-inf"))
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class _Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _Residual(_PreNorm(dim, _Attention(dim, heads=heads, dropout=dropout))),
                        _Residual(_PreNorm(dim, _FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for attn, ff in cast("list[tuple[nn.Module, nn.Module]]", self.layers):
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class _SeqTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_to_embedding = nn.Linear(channels * patch_size, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = _Transformer(dim, depth, heads, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_to_embedding(x)
        b = x.shape[0]
        c_tokens = repeat(self.c_token, "() n d -> b n d", b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        return x[:, 0]


# ---------------------------------------------------------------------------
# Public module
# ---------------------------------------------------------------------------


class TemporalContrast(nn.Module):
    """Temporal Contrastive module from TS-TCC (CPC-style).

    Computes a temporal contrastive (NCE) loss between two augmented views and
    returns a projection-head embedding for instance-level (NT-Xent) loss.
    """

    def __init__(self, num_channels: int, hidden_dim: int, timesteps: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.timestep = timesteps
        self.Wk = nn.ModuleList([nn.Linear(hidden_dim, num_channels) for _ in range(timesteps)])
        self.lsoftmax = nn.LogSoftmax(dim=-1)

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, num_channels // 2),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // 2, num_channels // 4),
        )
        self.seq_transformer = _SeqTransformer(
            patch_size=num_channels, dim=hidden_dim, depth=4, heads=4, mlp_dim=64
        )

    def forward(
        self, features_aug1: torch.Tensor, features_aug2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return temporal contrastive loss and projection for two views.

        Args:
            features_aug1: First feature view with shape ``(batch, num_channels, seq_len)``.
            features_aug2: Second feature view with shape ``(batch, num_channels, seq_len)``.

        Returns:
            nce:        scalar temporal contrastive loss
            projection: ``(batch, num_channels // 4)`` for NT-Xent
        """
        device = features_aug1.device
        z1 = features_aug1.transpose(1, 2)  # (batch, seq_len, num_channels)
        z2 = features_aug2.transpose(1, 2)

        batch, seq_len, _ = z1.shape
        if seq_len <= self.timestep:
            msg = f"seq_len ({seq_len}) must be > timestep ({self.timestep})"
            raise ValueError(msg)
        t_samples = torch.randint(seq_len - self.timestep, size=(1,), device=device).long()

        encode_samples = torch.stack(
            [
                z2[:, t_samples + i, :].view(batch, self.num_channels)
                for i in range(1, self.timestep + 1)
            ]
        )  # (timestep, batch, num_channels)

        c_t = self.seq_transformer(z1[:, : t_samples + 1, :])

        pred = torch.stack([self.Wk[i](c_t) for i in range(self.timestep)])

        nce = torch.tensor(0.0, device=device)
        for i in range(self.timestep):
            total = torch.mm(encode_samples[i], pred[i].T)  # (batch, batch)
            nce = nce + torch.sum(torch.diag(self.lsoftmax(total)))
        nce = -nce / (batch * self.timestep)

        return nce, self.projection_head(c_t)
