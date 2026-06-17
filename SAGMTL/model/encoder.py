# model/encoder.py
# -*- coding: utf-8 -*-
import math
import os
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .coblock import EdgeNodeCoBlock


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_mult: int = 2,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.drop = nn.Dropout(float(dropout))

        ff_dim = int(ff_mult) * self.hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(ff_dim, self.hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.drop(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))
        return x


class SpatialDistanceBiasNodeAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        exclude_self: int = 1,
        use_dist_decay: int = 1,
        dist_decay_init: float = 0.1,
        dist_decay_per_head: int = 0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.exclude_self = bool(int(exclude_self))
        self.use_dist_decay = bool(int(use_dist_decay))
        self.dist_decay_per_head = bool(int(dist_decay_per_head))

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}"
            )

        self.head_dim = self.hidden_dim // self.num_heads

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.drop = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.hidden_dim)

        init = torch.full(
            (self.num_heads,) if self.dist_decay_per_head else (),
            float(dist_decay_init),
            dtype=torch.float32,
        )
        self.gamma_logit = nn.Parameter(init)

    def forward(self, node_h: torch.Tensor, dist_bias: torch.Tensor) -> torch.Tensor:
        if node_h.dim() != 2:
            raise ValueError(f"node_h must be [N, H], got {tuple(node_h.shape)}")
        if dist_bias.dim() != 2:
            raise ValueError(f"dist_bias must be [N, N], got {tuple(dist_bias.shape)}")

        N, H = node_h.shape
        if dist_bias.size(0) != N or dist_bias.size(1) != N:
            raise ValueError(
                f"dist_bias must be [N, N], got {tuple(dist_bias.shape)} with N={N}"
            )

        if N <= 1:
            return torch.zeros_like(node_h)

        q = self.q_proj(node_h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(node_h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(node_h).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.use_dist_decay:
            gamma = F.softplus(self.gamma_logit)
            if self.dist_decay_per_head:
                score = score - gamma.view(self.num_heads, 1, 1) * dist_bias.unsqueeze(0)
            else:
                score = score - gamma * dist_bias.unsqueeze(0)

        if self.exclude_self:
            eye = torch.eye(N, device=node_h.device, dtype=torch.bool)
            score = score.masked_fill(eye.unsqueeze(0), float("-inf"))

        attn = torch.softmax(score, dim=-1)
        attn = self.drop(attn)

        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(0, 1).contiguous().view(N, H)
        ctx = self.out_proj(ctx)
        ctx = self.norm(ctx)
        return ctx


class TimeEdgeEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: Optional[int] = None,
        time_dim: int = 64,
        dropout: float = 0.1,
        use_edge_history: int = 1,
        edge_history_input: str = "flow_exist",
        edge_history_dim: int = 96,
        edge_history_chunk_size: int = 4096,
        edge_history_fuse: str = "concat",
        enc_dropout: Optional[float] = None,
        num_co_layers: int = 2,
        co_channels: int = 4,
        co_dropout: float = 0.1,
        use_directional_co: int = 0,
        use_time_dyn: int = 1,
        use_static_node: int = 1,
        static_weight: float = 0.5,
        dyn_weight: float = 0.5,
        normalize_blend: int = 1,
        static_dyn_fuse_mode: str = "concat",
        edge_chunk_size: int = 50000,
        use_time_attention: int = 0,
        time_attn_heads: int = 4,
        time_attn_layers: int = 1,
        time_attn_ff_mult: int = 2,
        time_attn_causal: int = 0,
        time_pool_mode: str = "last",
        use_spatial_attn: int = 1,
        spatial_dist_path: str = "",
        spatial_dist: Optional[Any] = None,
        spatial_attn_heads: int = 4,
        spatial_attn_dropout: float = 0.1,
        spatial_attn_res_weight: float = 0.15,
        spatial_attn_exclude_self: int = 1,
        use_spatial_dist_decay: int = 1,
        spatial_dist_decay_init: float = 0.1,
        spatial_dist_decay_per_head: int = 0,
        spatial_dist_norm: str = "max",
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim) if edge_dim is not None else int(hidden_dim)
        self.time_dim = int(time_dim)
        self.drop_p = float(dropout if enc_dropout is None else enc_dropout)

        self.edge_chunk_size = max(1, int(edge_chunk_size))

        self.use_time_dyn = bool(int(use_time_dyn))
        self.use_static_node = bool(int(use_static_node))
        self.static_weight = float(static_weight)
        self.dyn_weight = float(dyn_weight)
        self.normalize_blend = bool(int(normalize_blend))

        self.static_dyn_fuse_mode = str(static_dyn_fuse_mode or "concat").lower().strip()
        valid_fuse_modes = {"add", "blend", "weighted_sum", "concat", "cat", "concat_proj"}
        if self.static_dyn_fuse_mode not in valid_fuse_modes:
            raise ValueError(
                f"static_dyn_fuse_mode={self.static_dyn_fuse_mode!r} is invalid; "
                f"available modes are {sorted(valid_fuse_modes)}"
            )

        if (
            self.use_static_node
            and self.use_time_dyn
            and self.static_dyn_fuse_mode in {"add", "blend", "weighted_sum"}
        ):
            total_w = max(self.static_weight + self.dyn_weight, 1e-6)
            init_alpha = self.dyn_weight / total_w
            init_alpha = float(min(max(init_alpha, 1e-4), 1.0 - 1e-4))
            init_logit = math.log(init_alpha) - math.log(1.0 - init_alpha)
            self.blend_logit = nn.Parameter(torch.tensor([init_logit], dtype=torch.float32))
        else:
            self.blend_logit = None

        self.static_dyn_concat_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.static_dyn_concat_norm = nn.LayerNorm(self.hidden_dim)

        self.use_edge_history = bool(int(use_edge_history))
        self.edge_history_input = str(edge_history_input or "flow_exist").lower().strip()
        valid_edge_history_inputs = {"flow", "flow_exist", "raw_flow", "raw_flow_exist"}
        if self.edge_history_input not in valid_edge_history_inputs:
            raise ValueError(
                f"edge_history_input must be one of {sorted(valid_edge_history_inputs)}, "
                f"got {self.edge_history_input!r}"
            )

        self.edge_history_dim = int(edge_history_dim)
        self.edge_history_chunk_size = max(1, int(edge_history_chunk_size))
        self.edge_history_fuse = str(edge_history_fuse or "concat").lower().strip()
        if self.edge_history_fuse not in {"concat", "gate"}:
            raise ValueError(
                f"edge_history_fuse must be 'concat' or 'gate', got {self.edge_history_fuse!r}"
            )

        self.use_time_attention = bool(int(use_time_attention))
        self.time_attn_heads = int(time_attn_heads)
        self.time_attn_layers = int(time_attn_layers)
        self.time_attn_ff_mult = int(time_attn_ff_mult)
        self.time_attn_causal = bool(int(time_attn_causal))
        self.time_pool_mode = str(time_pool_mode or "last").lower().strip()
        if self.time_pool_mode not in {"last", "mean"}:
            raise ValueError(f"time_pool_mode must be 'last' or 'mean', got {self.time_pool_mode!r}")

        self.use_spatial_attn = bool(int(use_spatial_attn))
        self.spatial_attn_res_weight = float(spatial_attn_res_weight)
        self.spatial_attn_heads = int(spatial_attn_heads)
        self.spatial_attn_dropout = float(spatial_attn_dropout)
        self.spatial_attn_exclude_self = bool(int(spatial_attn_exclude_self))
        self.use_spatial_dist_decay = bool(int(use_spatial_dist_decay))
        self.spatial_dist_decay_init = float(spatial_dist_decay_init)
        self.spatial_dist_decay_per_head = bool(int(spatial_dist_decay_per_head))
        self.spatial_dist_norm = str(spatial_dist_norm or "max").lower().strip()
        if self.spatial_dist_norm not in {"none", "max", "log1p", "mean"}:
            raise ValueError(f"Unsupported spatial_dist_norm: {self.spatial_dist_norm}")

        if self.use_spatial_attn:
            self.spatial_attn = SpatialDistanceBiasNodeAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.spatial_attn_heads,
                dropout=self.spatial_attn_dropout,
                exclude_self=int(self.spatial_attn_exclude_self),
                use_dist_decay=int(self.use_spatial_dist_decay),
                dist_decay_init=self.spatial_dist_decay_init,
                dist_decay_per_head=int(self.spatial_dist_decay_per_head),
            )
        else:
            self.spatial_attn = None

        self.node_proj: Optional[nn.Linear] = None
        self.edge_proj: Optional[nn.Linear] = None

        self.time_proj = nn.Linear(self.time_dim, self.hidden_dim)
        self.temporal_gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        if self.use_time_attention:
            self.time_attn_stack = nn.ModuleList(
                [
                    TemporalSelfAttentionBlock(
                        hidden_dim=self.hidden_dim,
                        num_heads=self.time_attn_heads,
                        dropout=self.drop_p,
                        ff_mult=self.time_attn_ff_mult,
                    )
                    for _ in range(self.time_attn_layers)
                ]
            )
        else:
            self.time_attn_stack = None

        edge_hist_in_dim = 2 if self.edge_history_input in {"flow_exist", "raw_flow_exist"} else 1
        self.edge_hist_proj = nn.Linear(edge_hist_in_dim, self.edge_history_dim)
        self.edge_hist_gru = nn.GRU(
            input_size=self.edge_history_dim,
            hidden_size=self.edge_history_dim,
            batch_first=True,
        )
        self.edge_hist_to_hidden = nn.Linear(self.edge_history_dim, self.hidden_dim)
        self.edge_hist_norm = nn.LayerNorm(self.hidden_dim)
        self.edge_hist_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        self.norm_node = nn.LayerNorm(self.hidden_dim)
        self.prior_fuse_norm = nn.LayerNorm(self.hidden_dim)
        self.drop = nn.Dropout(self.drop_p)

        self.edge_mlp1: Optional[nn.Linear] = None
        self.edge_mlp2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.use_directional_co = bool(int(use_directional_co))
        self.num_co_layers = int(num_co_layers)
        if self.num_co_layers > 0:
            self.co_stack = nn.ModuleList(
                [
                    EdgeNodeCoBlock(
                        hidden_dim=self.hidden_dim,
                        channels=int(co_channels),
                        dropout=float(co_dropout),
                        use_directional_co=int(self.use_directional_co),
                    )
                    for _ in range(self.num_co_layers)
                ]
            )
        else:
            self.co_stack = None

        spatial_dist_tensor = self._load_spatial_dist(
            spatial_dist_path=spatial_dist_path,
            spatial_dist=spatial_dist,
        )
        if spatial_dist_tensor is None:
            spatial_dist_tensor = torch.empty(0, 0, dtype=torch.float32)
        self.register_buffer("spatial_dist_buf", spatial_dist_tensor, persistent=False)

    @staticmethod
    def _pick(batch: Dict[str, Any], candidates):
        for key in candidates:
            if key in batch and batch[key] is not None:
                return batch[key]
        return None

    @staticmethod
    def _load_spatial_dist(
        spatial_dist_path: str = "",
        spatial_dist: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        if spatial_dist is not None:
            return torch.as_tensor(spatial_dist, dtype=torch.float32)

        spatial_dist_path = str(spatial_dist_path or "").strip()
        if spatial_dist_path:
            if not os.path.isfile(spatial_dist_path):
                raise FileNotFoundError(f"spatial_dist_path does not exist: {spatial_dist_path}")
            arr = np.load(spatial_dist_path)
            return torch.as_tensor(arr, dtype=torch.float32)

        return None

    def _ensure_node_time_feat(self, batch: Dict[str, Any], N: int, device) -> torch.Tensor:
        node_time_feat = self._pick(batch, ["node_time_feat"])
        if node_time_feat is not None:
            return node_time_feat.to(device=device, dtype=torch.float32)

        time_scalar = self._pick(
            batch,
            ["time_id", "time_idx", "hour_id", "hour", "slot", "timeslot", "t", "t_idx"],
        )
        if time_scalar is None:
            t_val = torch.zeros(1, device=device, dtype=torch.float32)
        else:
            if not torch.is_tensor(time_scalar):
                t_val = torch.tensor([float(time_scalar)], device=device, dtype=torch.float32)
            else:
                t_val = time_scalar.to(device=device, dtype=torch.float32).view(-1)[:1]

        K = self.time_dim // 2
        w = 2.0 * math.pi * (t_val % 24.0) / 24.0
        ks = torch.arange(1, K + 1, device=device, dtype=torch.float32)

        feat = torch.cat([torch.sin(ks * w), torch.cos(ks * w)], dim=0)
        if feat.numel() < self.time_dim:
            pad = torch.zeros(self.time_dim - feat.numel(), device=device, dtype=torch.float32)
            feat = torch.cat([feat, pad], dim=0)

        feat = feat[: self.time_dim]
        return feat.view(1, 1, self.time_dim).expand(N, 1, self.time_dim).contiguous()

    def _lazy_linear(self, layer_ref_name: str, in_dim: int, device: torch.device) -> nn.Linear:
        layer = getattr(self, layer_ref_name)
        if layer is None:
            layer = nn.Linear(in_dim, self.hidden_dim).to(device)
            setattr(self, layer_ref_name, layer)
        return layer

    def _blend_static_dynamic(
        self,
        node_static: torch.Tensor,
        node_dyn: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_static_node and self.use_time_dyn:
            if self.static_dyn_fuse_mode in {"concat", "cat", "concat_proj"}:
                node_h = torch.cat([node_static, node_dyn], dim=-1)
                node_h = self.static_dyn_concat_proj(node_h)
                node_h = self.static_dyn_concat_norm(node_h)
            else:
                if self.blend_logit is not None:
                    alpha = torch.sigmoid(self.blend_logit)
                    node_h = (1.0 - alpha) * node_static + alpha * node_dyn
                else:
                    if self.normalize_blend:
                        s = max(self.static_weight, 0.0)
                        d = max(self.dyn_weight, 0.0)
                        z = max(s + d, 1e-6)
                        node_h = (s / z) * node_static + (d / z) * node_dyn
                    else:
                        node_h = self.static_weight * node_static + self.dyn_weight * node_dyn
        elif self.use_static_node:
            node_h = node_static
        else:
            node_h = node_dyn

        return node_h

    def _build_time_attn_mask(self, T: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.time_attn_causal:
            return None
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _encode_time_dynamic(
        self,
        time_h: torch.Tensor,
        node_static: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_time_dyn:
            return torch.zeros_like(node_static)

        if not self.use_time_attention:
            _, h_last = self.temporal_gru(time_h)
            return h_last[-1]

        attn_mask = self._build_time_attn_mask(T=time_h.size(1), device=time_h.device)

        h_seq = time_h
        for block in self.time_attn_stack:
            h_seq = block(h_seq, attn_mask=attn_mask)

        if self.time_pool_mode == "mean":
            return h_seq.mean(dim=1)
        return h_seq[:, -1, :]

    def _get_spatial_dist(
        self,
        batch: Dict[str, Any],
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        spatial_dist = self._pick(batch, ["spatial_dist", "spatial_dist_bias"])
        if spatial_dist is None:
            spatial_dist = self.spatial_dist_buf

        if spatial_dist is None or spatial_dist.numel() == 0:
            raise RuntimeError(
                "use_spatial_attn=1 requires spatial_dist. "
                "Pass spatial_dist_path during initialization or provide batch['spatial_dist']."
            )

        spatial_dist = spatial_dist.to(device=device, dtype=torch.float32)
        if spatial_dist.dim() != 2 or spatial_dist.size(0) != N or spatial_dist.size(1) != N:
            raise ValueError(
                f"spatial_dist must be [N, N], got {tuple(spatial_dist.shape)} with N={N}"
            )

        if not torch.isfinite(spatial_dist).all():
            raise ValueError("spatial_dist contains NaN or Inf values")

        return self._normalize_spatial_dist(spatial_dist)

    def _normalize_spatial_dist(self, spatial_dist: torch.Tensor) -> torch.Tensor:
        mode = self.spatial_dist_norm
        D = spatial_dist

        if mode == "none":
            return D

        if mode == "log1p":
            D = torch.log1p(torch.clamp(D, min=0.0))
            return D / D.max().clamp_min(1e-6)

        if mode == "max":
            return D / D.max().clamp_min(1e-6)

        if mode == "mean":
            pos = D[D > 0]
            denom = pos.mean() if pos.numel() > 0 else D.new_tensor(1.0)
            return D / denom.clamp_min(1e-6)

        raise ValueError(f"Unsupported spatial_dist_norm: {mode}")

    def _encode_edge_history(
        self,
        batch: Dict[str, Any],
        E: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.use_edge_history:
            return None

        flow_hist = self._pick(batch, ["edge_flow_hist", "edge_hist_flow", "edge_history_flow"])
        if flow_hist is None:
            return torch.zeros((E, self.hidden_dim), device=device, dtype=dtype)

        flow_hist = flow_hist.to(device=device, dtype=torch.float32)

        if flow_hist.dim() == 3 and flow_hist.size(-1) == 1:
            flow_hist = flow_hist.squeeze(-1)

        if flow_hist.dim() != 2:
            raise ValueError(f"edge_flow_hist must be [E, T], got {tuple(flow_hist.shape)}")

        if flow_hist.size(0) != E:
            raise ValueError(
                f"edge_flow_hist first dimension must match E={E}, got {tuple(flow_hist.shape)}"
            )

        use_exist_channel = self.edge_history_input in {"flow_exist", "raw_flow_exist"}
        use_raw_flow = self.edge_history_input in {"raw_flow", "raw_flow_exist"}

        exist_hist_all = None
        if use_exist_channel:
            exist_hist_all = self._pick(batch, ["edge_exist_hist", "edge_hist_exist", "edge_history_exist"])
            if exist_hist_all is not None:
                exist_hist_all = exist_hist_all.to(device=device, dtype=torch.float32)
                if exist_hist_all.dim() == 3 and exist_hist_all.size(-1) == 1:
                    exist_hist_all = exist_hist_all.squeeze(-1)
                if exist_hist_all.dim() != 2 or exist_hist_all.size(0) != E:
                    raise ValueError(
                        f"edge_exist_hist must be [E, T] with E={E}, got {tuple(exist_hist_all.shape)}"
                    )

        hist_chunks = []
        chunk_size = max(1, int(self.edge_history_chunk_size))

        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)

            flow_chunk = flow_hist[start:end]
            flow_nonneg = torch.clamp(flow_chunk, min=0.0)

            if use_raw_flow:
                x_flow = flow_nonneg.unsqueeze(-1)
            else:
                x_flow = torch.log1p(flow_nonneg).unsqueeze(-1)

            if use_exist_channel:
                if exist_hist_all is None:
                    x_exist = (flow_chunk > 0).float().unsqueeze(-1)
                else:
                    x_exist = exist_hist_all[start:end].unsqueeze(-1)
                x = torch.cat([x_flow, x_exist], dim=-1)
            else:
                x = x_flow

            x = torch.relu(self.edge_hist_proj(x))
            _, h_last = self.edge_hist_gru(x)

            hist = self.edge_hist_to_hidden(h_last[-1])
            hist = self.edge_hist_norm(hist).to(dtype=dtype)
            hist_chunks.append(hist)

        if not hist_chunks:
            return torch.empty((0, self.hidden_dim), device=device, dtype=dtype)

        return torch.cat(hist_chunks, dim=0)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        node_feat = self._pick(batch, ["node_feat", "node_features", "node_attr"])
        if node_feat is None:
            raise KeyError("TimeEdgeEncoder requires node features: node_feat / node_features / node_attr")
        node_feat = node_feat.to(device=device, dtype=torch.float32)
        N = node_feat.size(0)

        edge_index = self._pick(batch, ["edge_index"])
        if edge_index is None:
            raise KeyError("TimeEdgeEncoder requires edge_index with shape [2, E]")
        edge_index = edge_index.to(device=device, dtype=torch.long)

        src, dst = edge_index[0], edge_index[1]
        E = src.numel()

        node_fc = self._lazy_linear("node_proj", node_feat.size(-1), device)
        node_static = node_fc(node_feat)

        node_time_feat = self._ensure_node_time_feat(batch, N, device)
        time_h = self.time_proj(node_time_feat)
        time_h = self.drop(time_h)

        node_dyn = self._encode_time_dynamic(time_h, node_static)

        node_h = self._blend_static_dynamic(node_static, node_dyn)
        node_h = self.drop(self.norm_node(node_h))

        last_flow = self._pick(batch, ["last_flow", "prev_flow", "flow_tminus1"])
        if last_flow is not None:
            last_flow = last_flow.to(device=device, dtype=torch.float32)
            if last_flow.dim() == 2 and last_flow.size(-1) == 1:
                last_flow = last_flow.squeeze(-1)

        spatial_msg = None
        if self.use_spatial_attn:
            spatial_dist = self._get_spatial_dist(batch=batch, N=N, device=device)
            spatial_msg = self.spatial_attn(node_h=node_h, dist_bias=spatial_dist)
            node_h = node_h + self.spatial_attn_res_weight * self.drop(spatial_msg)
            node_h = self.drop(self.prior_fuse_norm(node_h))

        edge_feat = self._pick(batch, ["edge_feat", "edge_attr"])
        if edge_feat is not None:
            edge_feat = edge_feat.to(device=device, dtype=torch.float32)
            edge_fc = self._lazy_linear("edge_proj", edge_feat.size(-1), device)
            e_static_all = edge_fc(edge_feat)
        else:
            e_static_all = None

        edge_hist_all = self._encode_edge_history(
            batch=batch,
            E=E,
            device=device,
            dtype=node_h.dtype,
        )

        use_edge_hist_concat = edge_hist_all is not None and self.edge_history_fuse == "concat"
        edge_input_dim = self.hidden_dim * (4 if use_edge_hist_concat else 3)

        if self.edge_mlp1 is None:
            self.edge_mlp1 = nn.Linear(edge_input_dim, self.hidden_dim).to(device)

        chunk_size = int(self.edge_chunk_size) if self.edge_chunk_size is not None else E
        if chunk_size <= 0:
            chunk_size = E

        edge_h_chunks = []

        for start in range(0, E, chunk_size):
            end = min(E, start + chunk_size)

            s_idx = src[start:end]
            d_idx = dst[start:end]

            h_src = node_h.index_select(0, s_idx)
            h_dst = node_h.index_select(0, d_idx)

            if e_static_all is not None:
                e_static = e_static_all[start:end]
            else:
                e_static = torch.zeros(
                    h_src.size(0),
                    self.hidden_dim,
                    device=device,
                    dtype=node_h.dtype,
                )

            if use_edge_hist_concat:
                e_hist = edge_hist_all[start:end]
                edge_in = torch.cat([h_src, h_dst, e_static, e_hist], dim=-1)
            else:
                edge_in = torch.cat([h_src, h_dst, e_static], dim=-1)

            edge_h_chunk = F.relu(self.edge_mlp1(edge_in))
            edge_h_chunk = self.drop(self.edge_mlp2(edge_h_chunk))

            if edge_hist_all is not None and self.edge_history_fuse == "gate":
                e_hist = edge_hist_all[start:end]
                gate = self.edge_hist_gate(torch.cat([edge_h_chunk, e_hist], dim=-1))
                edge_h_chunk = (1.0 - gate) * edge_h_chunk + gate * e_hist

            edge_h_chunks.append(edge_h_chunk)

        if edge_h_chunks:
            edge_h = torch.cat(edge_h_chunks, dim=0)
        else:
            edge_h = torch.empty(
                0,
                self.hidden_dim,
                device=device,
                dtype=node_h.dtype,
            )

        if self.co_stack is not None:
            for block in self.co_stack:
                node_h, edge_h = block(node_h, edge_h, edge_index, num_nodes=N)

        out = {
            "node_h": node_h,
            "edge_h": edge_h,
            "node_static": node_static,
            "node_dyn": node_dyn,
        }

        if edge_hist_all is not None:
            out["edge_hist"] = edge_hist_all

        if spatial_msg is not None:
            out["spatial_msg"] = spatial_msg
            if self.spatial_attn is not None:
                out["spatial_gamma"] = F.softplus(self.spatial_attn.gamma_logit).detach()

        if last_flow is not None:
            out["last_flow"] = last_flow

        return out