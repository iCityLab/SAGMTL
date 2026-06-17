# model/decoder.py
# -*- coding: utf-8 -*-
import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelMLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RoleProjection(nn.Module):
    def __init__(self, in_dim: int, role_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, role_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QueryCrossAttentionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        horizon: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_mult: int = 2,
        num_memory_tokens: int = 4,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.num_heads = int(num_heads)
        self.num_memory_tokens = max(1, int(num_memory_tokens))

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}"
            )

        self.input_norm = nn.LayerNorm(self.in_dim)

        self.backbone = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.global_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.memory_proj = nn.Linear(self.hidden_dim, self.num_memory_tokens * self.hidden_dim)

        self.memory_norm1 = nn.LayerNorm(self.hidden_dim)
        self.memory_norm2 = nn.LayerNorm(self.hidden_dim)

        mem_ff_dim = int(ff_mult) * self.hidden_dim
        self.memory_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, mem_ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(mem_ff_dim, self.hidden_dim),
        )

        self.horizon_queries = nn.Parameter(
            torch.randn(1, self.horizon, self.hidden_dim) * 0.02
        )

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

        self.out_proj = nn.Linear(self.hidden_dim, 1)
        self.skip_proj = nn.Linear(self.hidden_dim, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be [B, F], got {tuple(x.shape)}")
        if x.size(-1) != self.in_dim:
            raise ValueError(f"x last dim must be {self.in_dim}, got {x.size(-1)}")

        B = x.size(0)

        h0 = self.backbone(self.input_norm(x))
        global_tok = self.global_proj(h0).unsqueeze(1)
        latent = self.memory_proj(h0).view(
            B,
            self.num_memory_tokens,
            self.hidden_dim,
        )

        memory = torch.cat([global_tok, latent], dim=1)
        memory = self.memory_norm1(memory)
        memory = self.memory_norm2(memory + self.drop(self.memory_ffn(memory)))

        q = self.horizon_queries.expand(B, -1, -1)
        attn_out, _ = self.attn(q, memory, memory, need_weights=False)

        h = self.norm1(q + self.drop(attn_out))
        h = self.norm2(h + self.drop(self.ffn(h)))

        y = self.out_proj(h).squeeze(-1)
        y = y + self.skip_proj(h0)
        return y


class StaticBasisHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        horizon: int,
        rank: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.rank = max(1, int(rank))

        self.input_norm = nn.LayerNorm(self.in_dim)

        self.backbone = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )

        self.coeff_proj = nn.Linear(self.hidden_dim, self.rank)
        self.gate_proj = nn.Linear(self.hidden_dim, self.rank)

        self.bias = nn.Parameter(torch.zeros(self.horizon))
        self.temporal_basis = nn.Parameter(torch.empty(self.rank, self.horizon))

        self.residual_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.horizon),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            t = torch.linspace(0.0, 1.0, self.horizon)
            patterns = [
                torch.ones_like(t),
                t,
                1.0 - t,
                (t - 0.5) ** 2,
                torch.sin(math.pi * t),
                torch.cos(math.pi * t),
            ]

            self.temporal_basis.zero_()
            for i in range(self.rank):
                if i < len(patterns):
                    self.temporal_basis[i].copy_(patterns[i])
                else:
                    self.temporal_basis[i].normal_(mean=0.0, std=0.02)

            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be [B, F], got {tuple(x.shape)}")
        if x.size(-1) != self.in_dim:
            raise ValueError(f"x last dim must be {self.in_dim}, got {x.size(-1)}")

        h = self.backbone(self.input_norm(x))

        coeff = self.coeff_proj(h)
        gate = torch.sigmoid(self.gate_proj(h))
        coeff = coeff * gate

        main = coeff @ self.temporal_basis + self.bias
        residual = self.residual_head(h)
        return main + self.residual_scale * residual


class ScalarConditionedGate(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_scalar: int,
        hidden_dim: int,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.scalar_proj = nn.Linear(num_scalar, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.use_sigmoid = bool(use_sigmoid)

    def forward(self, feat: torch.Tensor, scalar_seq: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 2:
            raise ValueError(f"feat must be [B, F], got {tuple(feat.shape)}")
        if scalar_seq.dim() != 3:
            raise ValueError(f"scalar_seq must be [B, H, S], got {tuple(scalar_seq.shape)}")

        B, H, _ = scalar_seq.shape

        feat_h = self.feat_proj(feat).unsqueeze(1)
        scalar_h = self.scalar_proj(scalar_seq.reshape(B * H, -1)).view(B, H, -1)

        h = F.relu(feat_h + scalar_h)
        out = self.out(h).squeeze(-1)

        if self.use_sigmoid:
            out = torch.sigmoid(out)

        return out


class MultiTaskGatedDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: Optional[int] = None,
        horizon: int = 1,
        role_dim: Optional[int] = None,
        use_gate1: bool = True,
        use_sd_gate: bool = True,
        fixed_alpha: float = 0.7,
        use_od_gate: bool = True,
        gate2_detach: bool = True,
        use_last_flow_gate: bool = True,
        use_residual_bias: bool = True,
        use_single_head_fusion: bool = False,
        dyn_decoder_mode: str = "residual",
        dyn_res_use_tanh: bool = True,
        dyn_res_scale: float = 1.0,
        use_edge_history_in_decoder: bool = True,
        edge_history_decoder_detach: bool = False,
        static_branch_mode: str = "static_roles",
        sd_fusion_mode: str = "adaptive",
        use_static_basis_head: bool = False,
        static_basis_rank: int = 4,
        static_basis_dropout: Optional[float] = None,
        use_dyn_query_head: bool = False,
        dyn_query_heads: int = 4,
        dyn_query_ff_mult: int = 2,
        dyn_query_dropout: Optional[float] = None,
        dyn_query_tokens: int = 4,
        edge_chunk_size: int = 1024,
        dropout: float = 0.1,
        max_log_gain: float = 2.5,
        dyn_share_floor: float = 0.35,
        static_scale: float = 0.60,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim) if edge_dim is not None else int(hidden_dim)
        self.horizon = int(horizon)
        self.role_dim = int(role_dim) if role_dim is not None else int(hidden_dim)

        self.use_gate1 = bool(use_gate1)
        self.use_sd_gate = bool(use_sd_gate)
        self.fixed_alpha = float(fixed_alpha)
        self.use_od_gate = bool(use_od_gate)
        self.gate2_detach = bool(gate2_detach)
        self.use_last_flow_gate = bool(use_last_flow_gate)
        self.use_residual_bias = bool(use_residual_bias)
        self.use_single_head_fusion = bool(use_single_head_fusion)

        self.dyn_decoder_mode = str(dyn_decoder_mode or "residual").lower().strip()
        if self.dyn_decoder_mode not in {"residual", "exp_clip"}:
            raise ValueError(
                f"dyn_decoder_mode must be 'residual' or 'exp_clip', got {self.dyn_decoder_mode}"
            )

        self.dyn_res_use_tanh = bool(dyn_res_use_tanh)
        self.dyn_res_scale = float(dyn_res_scale)
        self.max_log_gain = max(0.25, float(max_log_gain))

        self.use_edge_history_in_decoder = bool(use_edge_history_in_decoder)
        self.edge_history_decoder_detach = bool(edge_history_decoder_detach)

        self.static_branch_mode = str(static_branch_mode or "static_roles").lower().strip()
        if self.static_branch_mode not in {"co", "static_roles"}:
            raise ValueError(
                f"static_branch_mode must be 'co' or 'static_roles', got {self.static_branch_mode}"
            )

        self.sd_fusion_mode = str(sd_fusion_mode or "adaptive").lower().strip()
        valid_sd_fusion_modes = {
            "adaptive",
            "no_floor",
            "fixed",
            "simple_learned",
            "struct_dyn",
        }
        if self.sd_fusion_mode not in valid_sd_fusion_modes:
            raise ValueError(
                f"sd_fusion_mode={self.sd_fusion_mode!r} is invalid; "
                f"available modes are {sorted(valid_sd_fusion_modes)}"
            )

        self.use_static_basis_head = bool(use_static_basis_head)
        self.static_basis_rank = max(1, int(static_basis_rank))
        self.static_basis_dropout = float(
            dropout if static_basis_dropout is None else static_basis_dropout
        )

        self.use_dyn_query_head = bool(use_dyn_query_head)
        self.dyn_query_heads = int(dyn_query_heads)
        self.dyn_query_ff_mult = int(dyn_query_ff_mult)
        self.dyn_query_dropout = float(
            dropout if dyn_query_dropout is None else dyn_query_dropout
        )
        self.dyn_query_tokens = max(1, int(dyn_query_tokens))

        self.edge_chunk_size = max(1, int(edge_chunk_size))
        self.dyn_share_floor = max(0.05, min(float(dyn_share_floor), 0.95))
        self.static_scale = max(0.05, float(static_scale))

        self.norm_e = nn.LayerNorm(self.hidden_dim)
        self.norm_n = nn.LayerNorm(self.hidden_dim)
        self.norm_ns = nn.LayerNorm(self.hidden_dim)
        self.norm_nd = nn.LayerNorm(self.hidden_dim)
        self.drop = nn.Dropout(p=dropout)

        self.out_role_head = RoleProjection(self.hidden_dim, self.role_dim, dropout=dropout)
        self.in_role_head = RoleProjection(self.hidden_dim, self.role_dim, dropout=dropout)

        self.out_role_static = RoleProjection(self.hidden_dim, self.role_dim, dropout=dropout)
        self.in_role_static = RoleProjection(self.hidden_dim, self.role_dim, dropout=dropout)

        self.out_role_dyn = RoleProjection(self.hidden_dim, self.role_dim, dropout=dropout)
        self.in_role_dyn = RoleProjection(self.hidden_dim, self.role_dim, dropout=dropout)

        self.node_out_pred = nn.Sequential(
            nn.Linear(self.role_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_dim, self.horizon),
        )
        self.node_in_pred = nn.Sequential(
            nn.Linear(self.role_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_dim, self.horizon),
        )

        exist_in_dim = self.hidden_dim + 2 * self.role_dim

        static_flow_in_dim = (
            2 * self.role_dim
            if self.static_branch_mode == "static_roles"
            else self.hidden_dim + 2 * self.role_dim
        )

        dyn_flow_in_dim = self.hidden_dim + 2 * self.role_dim
        if self.use_edge_history_in_decoder:
            dyn_flow_in_dim += self.hidden_dim

        fused_in_dim = self.hidden_dim + 4 * self.role_dim
        if self.use_edge_history_in_decoder:
            fused_in_dim += self.hidden_dim

        self.od_head = ParallelMLPHead(
            exist_in_dim,
            self.hidden_dim,
            self.horizon,
            dropout=dropout,
        )

        if self.use_static_basis_head:
            self.flow_static_head = StaticBasisHead(
                in_dim=static_flow_in_dim,
                hidden_dim=self.hidden_dim,
                horizon=self.horizon,
                rank=self.static_basis_rank,
                dropout=self.static_basis_dropout,
            )
        else:
            self.flow_static_head = ParallelMLPHead(
                static_flow_in_dim,
                self.hidden_dim,
                self.horizon,
                dropout=dropout,
            )

        if self.use_dyn_query_head:
            self.flow_dyn_level_head = QueryCrossAttentionHead(
                in_dim=dyn_flow_in_dim,
                hidden_dim=self.hidden_dim,
                horizon=self.horizon,
                num_heads=self.dyn_query_heads,
                dropout=self.dyn_query_dropout,
                ff_mult=self.dyn_query_ff_mult,
                num_memory_tokens=self.dyn_query_tokens,
            )
            self.flow_dyn_gain_head = QueryCrossAttentionHead(
                in_dim=dyn_flow_in_dim,
                hidden_dim=self.hidden_dim,
                horizon=self.horizon,
                num_heads=self.dyn_query_heads,
                dropout=self.dyn_query_dropout,
                ff_mult=self.dyn_query_ff_mult,
                num_memory_tokens=self.dyn_query_tokens,
            )
            self.flow_fused_head = (
                QueryCrossAttentionHead(
                    in_dim=fused_in_dim,
                    hidden_dim=self.hidden_dim,
                    horizon=self.horizon,
                    num_heads=self.dyn_query_heads,
                    dropout=self.dyn_query_dropout,
                    ff_mult=self.dyn_query_ff_mult,
                    num_memory_tokens=self.dyn_query_tokens,
                )
                if self.use_single_head_fusion
                else None
            )
        else:
            self.flow_dyn_level_head = ParallelMLPHead(
                dyn_flow_in_dim,
                self.hidden_dim,
                self.horizon,
                dropout=dropout,
            )
            self.flow_dyn_gain_head = ParallelMLPHead(
                dyn_flow_in_dim,
                self.hidden_dim,
                self.horizon,
                dropout=dropout,
            )
            self.flow_fused_head = (
                ParallelMLPHead(
                    fused_in_dim,
                    self.hidden_dim,
                    self.horizon,
                    dropout=dropout,
                )
                if self.use_single_head_fusion
                else None
            )

        self.simple_alpha_head = None
        if self.sd_fusion_mode == "simple_learned":
            self.simple_alpha_head = ParallelMLPHead(
                fused_in_dim,
                self.hidden_dim,
                self.horizon,
                dropout=dropout,
            )

        self.struct_dyn_alpha_head = None
        if self.sd_fusion_mode == "struct_dyn":
            self.struct_dyn_alpha_head = nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(self.hidden_dim, 1),
            )

        self.gate1 = None
        if self.use_gate1:
            self.gate1 = ScalarConditionedGate(
                dyn_flow_in_dim,
                1,
                self.hidden_dim,
                use_sigmoid=True,
            )

        self.gate2 = None
        if self.use_sd_gate:
            self.gate2 = ScalarConditionedGate(
                exist_in_dim,
                2,
                self.hidden_dim,
                use_sigmoid=False,
            )

    def _get_last_flow_vector(
        self,
        enc_out: Dict[str, torch.Tensor],
        batch: Optional[Dict[str, Any]],
        edge_repr: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_last_flow_gate:
            return None

        last_flow_all = None
        if enc_out is not None:
            if "last_flow" in enc_out:
                last_flow_all = enc_out["last_flow"]
            elif "edge_last_flow" in enc_out:
                last_flow_all = enc_out["edge_last_flow"]

        if last_flow_all is None and batch is not None:
            if "last_flow" in batch:
                last_flow_all = batch["last_flow"]
            elif "edge_last_flow" in batch:
                last_flow_all = batch["edge_last_flow"]

        if last_flow_all is None:
            return None

        if not isinstance(last_flow_all, torch.Tensor):
            last_flow_all = torch.as_tensor(
                last_flow_all,
                device=edge_repr.device,
                dtype=edge_repr.dtype,
            )
        else:
            last_flow_all = last_flow_all.to(edge_repr.device, dtype=edge_repr.dtype)

        last_flow_all = last_flow_all.view(-1)
        if last_flow_all.shape[0] != edge_repr.size(0):
            return None

        return last_flow_all

    def _format_node_output(self, node_logit_raw: torch.Tensor) -> torch.Tensor:
        if self.horizon == 1:
            return node_logit_raw[:, 0, :].contiguous()
        return node_logit_raw.permute(1, 0, 2).contiguous()

    def _format_edge_output(self, edge_x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if edge_x is None:
            return None
        if self.horizon == 1:
            return edge_x[:, 0].contiguous()
        return edge_x.transpose(0, 1).contiguous()

    def _build_node_roles(
        self,
        node_repr: torch.Tensor,
        node_static: torch.Tensor,
        node_dyn: torch.Tensor,
    ):
        n = self.drop(self.norm_n(node_repr))
        ns = self.drop(self.norm_ns(node_static))
        nd = self.drop(self.norm_nd(node_dyn))

        out_role = self.out_role_head(n)
        in_role = self.in_role_head(n)

        out_role_s = self.out_role_static(ns)
        in_role_s = self.in_role_static(ns)

        out_role_d = self.out_role_dyn(nd)
        in_role_d = self.in_role_dyn(nd)

        node_out_raw = self.node_out_pred(out_role)
        node_in_raw = self.node_in_pred(in_role)

        node_logit_raw = torch.stack([node_out_raw, node_in_raw], dim=-1)

        return (
            out_role,
            in_role,
            node_logit_raw,
            out_role_s,
            in_role_s,
            out_role_d,
            in_role_d,
        )

    def _compute_alpha_dyn(
        self,
        exist_in: torch.Tensor,
        fused_in: torch.Tensor,
        edge_logit: torch.Tensor,
        g1: torch.Tensor,
        dyn_ref: torch.Tensor,
    ) -> torch.Tensor:
        if self.sd_fusion_mode == "fixed":
            return torch.full_like(dyn_ref, fill_value=self.fixed_alpha)

        if self.sd_fusion_mode == "simple_learned":
            if self.simple_alpha_head is None:
                raise RuntimeError(
                    "sd_fusion_mode='simple_learned' but simple_alpha_head is not initialized"
                )

            alpha_raw = self.simple_alpha_head(fused_in)
            alpha = torch.sigmoid(alpha_raw)
            return self.dyn_share_floor + (1.0 - self.dyn_share_floor) * alpha

        if self.sd_fusion_mode == "struct_dyn":
            if self.struct_dyn_alpha_head is None:
                raise RuntimeError(
                    "sd_fusion_mode='struct_dyn' but struct_dyn_alpha_head is not initialized"
                )

            if self.use_od_gate:
                od_in = edge_logit.detach() if self.gate2_detach else edge_logit
                g1_in = g1.detach() if self.gate2_detach else g1
            else:
                od_in = torch.zeros_like(edge_logit)
                g1_in = torch.zeros_like(g1)

            scalar_seq = torch.stack([od_in, g1_in], dim=-1)
            alpha_raw = self.struct_dyn_alpha_head(scalar_seq).squeeze(-1)
            return torch.sigmoid(alpha_raw)

        if (not self.use_sd_gate) or (self.gate2 is None):
            raise RuntimeError(
                f"sd_fusion_mode='{self.sd_fusion_mode}' requires gate2; "
                "set use_sd_gate=True or use fixed/simple_learned/struct_dyn"
            )

        if self.use_od_gate:
            od_in = edge_logit.detach() if self.gate2_detach else edge_logit
            g1_in = g1.detach() if self.gate2_detach else g1
        else:
            od_in = torch.zeros_like(edge_logit)
            g1_in = torch.zeros_like(g1)

        scalar_seq = torch.stack([g1_in, od_in], dim=-1)
        alpha = torch.sigmoid(self.gate2(exist_in, scalar_seq))

        if self.sd_fusion_mode == "adaptive":
            return self.dyn_share_floor + (1.0 - self.dyn_share_floor) * alpha

        if self.sd_fusion_mode == "no_floor":
            return alpha

        raise RuntimeError(f"Unsupported sd_fusion_mode={self.sd_fusion_mode!r}")

    def forward(
        self,
        enc_out: Dict[str, torch.Tensor],
        batch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        if "edge_h" not in enc_out or "node_h" not in enc_out:
            raise KeyError("decoder expects enc_out to contain 'edge_h' and 'node_h'")
        if "node_static" not in enc_out or "node_dyn" not in enc_out:
            raise KeyError("decoder expects enc_out to contain 'node_static' and 'node_dyn'")
        if batch is None or "edge_index" not in batch:
            raise KeyError("decoder requires batch['edge_index'] aligned with edge_h")

        edge_repr = enc_out["edge_h"]
        node_repr = enc_out["node_h"]
        node_static = enc_out["node_static"]
        node_dyn = enc_out["node_dyn"]

        edge_index = batch["edge_index"]
        src_all = edge_index[0].long()
        dst_all = edge_index[1].long()

        E = edge_repr.size(0)
        if src_all.numel() != E:
            raise ValueError(
                f"edge_h.shape[0]={E} does not match edge_index.size(1)={src_all.numel()}"
            )

        (
            out_role,
            in_role,
            node_logit_raw,
            out_role_s,
            in_role_s,
            out_role_d,
            in_role_d,
        ) = self._build_node_roles(
            node_repr,
            node_static,
            node_dyn,
        )

        node_logit = self._format_node_output(node_logit_raw)

        if E == 0:
            device = edge_repr.device
            dtype = edge_repr.dtype
            empty_edge = torch.empty((0, self.horizon), device=device, dtype=dtype)

            return {
                "flow_pred": self._format_edge_output(empty_edge),
                "flow_cond": self._format_edge_output(empty_edge),
                "flow_static": self._format_edge_output(empty_edge),
                "flow_dyn": self._format_edge_output(empty_edge),
                "edge_prob": self._format_edge_output(empty_edge),
                "alpha_sd": self._format_edge_output(empty_edge),
                "g1": self._format_edge_output(empty_edge),
                "od_edge_logits": self._format_edge_output(empty_edge),
                "onset_logit": None,
                "burst_logit": None,
                "od_node_logits": node_logit,
            }

        last_flow_all = self._get_last_flow_vector(enc_out, batch, edge_repr)

        edge_hist_all = None
        if self.use_edge_history_in_decoder:
            edge_hist_all = enc_out.get("edge_hist", None)
            if edge_hist_all is None:
                raise KeyError("use_edge_history_in_decoder=True requires enc_out['edge_hist']")

            edge_hist_all = edge_hist_all.to(device=edge_repr.device, dtype=edge_repr.dtype)

            if (
                edge_hist_all.dim() != 2
                or edge_hist_all.size(0) != E
                or edge_hist_all.size(1) != self.hidden_dim
            ):
                raise ValueError(
                    f"edge_hist must be [E, {self.hidden_dim}] with E={E}, "
                    f"got {tuple(edge_hist_all.shape)}"
                )

            if self.edge_history_decoder_detach:
                edge_hist_all = edge_hist_all.detach()

        flow_final_chunks = []
        flow_cond_chunks = []
        flow_s_chunks = []
        flow_d_chunks = []
        edge_prob_chunks = []
        alpha_dyn_chunks = []
        g1_chunks = []
        edge_logit_chunks = []

        for start in range(0, E, self.edge_chunk_size):
            end = min(E, start + self.edge_chunk_size)

            e = self.drop(self.norm_e(edge_repr[start:end]))

            src = src_all[start:end]
            dst = dst_all[start:end]

            out_src = out_role.index_select(0, src)
            in_dst = in_role.index_select(0, dst)

            out_src_s = out_role_s.index_select(0, src)
            in_dst_s = in_role_s.index_select(0, dst)

            out_src_d = out_role_d.index_select(0, src)
            in_dst_d = in_role_d.index_select(0, dst)

            exist_in = torch.cat([e, out_src, in_dst], dim=-1)

            if self.static_branch_mode == "static_roles":
                static_in = torch.cat([out_src_s, in_dst_s], dim=-1)
            else:
                static_in = torch.cat([e, out_src_s, in_dst_s], dim=-1)

            edge_hist_chunk = edge_hist_all[start:end] if edge_hist_all is not None else None

            dyn_parts = [e, out_src_d, in_dst_d]
            if edge_hist_chunk is not None:
                dyn_parts.append(edge_hist_chunk)
            dyn_in = torch.cat(dyn_parts, dim=-1)

            fused_parts = [e, out_src_s, in_dst_s, out_src_d, in_dst_d]
            if edge_hist_chunk is not None:
                fused_parts.append(edge_hist_chunk)
            fused_in = torch.cat(fused_parts, dim=-1)

            edge_logit = self.od_head(exist_in)
            edge_prob = torch.sigmoid(edge_logit)

            if self.use_single_head_fusion:
                if self.flow_fused_head is None:
                    raise RuntimeError(
                        "use_single_head_fusion=True but flow_fused_head is not initialized"
                    )

                flow_cond = F.softplus(self.flow_fused_head(fused_in))
                flow_static = torch.zeros_like(flow_cond)
                flow_dyn = flow_cond
                alpha_dyn = torch.ones_like(flow_cond)
                g1 = torch.ones_like(flow_cond)

            else:
                flow_static_raw = self.flow_static_head(static_in)
                flow_static = self.static_scale * F.softplus(flow_static_raw)

                dyn_level_raw = self.flow_dyn_level_head(dyn_in)
                flow_dyn_level = F.softplus(dyn_level_raw)

                dyn_gain_raw = self.flow_dyn_gain_head(dyn_in)

                if self.use_gate1 and self.gate1 is not None:
                    if self.use_last_flow_gate and last_flow_all is not None:
                        gate_feature = last_flow_all[start:end].unsqueeze(1).expand(
                            -1,
                            self.horizon,
                        )
                    else:
                        gate_feature = flow_dyn_level.detach()

                    g1 = self.gate1(dyn_in, gate_feature.unsqueeze(-1))
                else:
                    g1 = torch.ones_like(dyn_gain_raw)

                if self.dyn_res_use_tanh:
                    dyn_residual = g1 * torch.tanh(dyn_gain_raw)
                else:
                    dyn_residual = g1 * dyn_gain_raw

                if self.dyn_decoder_mode == "residual":
                    flow_dyn_main = F.softplus(
                        dyn_level_raw + self.dyn_res_scale * dyn_residual
                    )
                else:
                    log_gain = torch.clamp(
                        dyn_residual,
                        min=-self.max_log_gain,
                        max=self.max_log_gain,
                    )
                    flow_dyn_main = flow_dyn_level * torch.exp(log_gain)

                alpha_dyn = self._compute_alpha_dyn(
                    exist_in=exist_in,
                    fused_in=fused_in,
                    edge_logit=edge_logit,
                    g1=g1,
                    dyn_ref=dyn_gain_raw,
                )

                flow_cond = (1.0 - alpha_dyn) * flow_static + alpha_dyn * flow_dyn_main
                flow_dyn = F.relu(flow_cond - flow_static)

            flow_final = flow_cond

            flow_final_chunks.append(flow_final)
            flow_cond_chunks.append(flow_cond)
            flow_s_chunks.append(flow_static)
            flow_d_chunks.append(flow_dyn)
            edge_prob_chunks.append(edge_prob)
            alpha_dyn_chunks.append(alpha_dyn)
            g1_chunks.append(g1)
            edge_logit_chunks.append(edge_logit)

        flow_final = torch.cat(flow_final_chunks, dim=0)
        flow_cond = torch.cat(flow_cond_chunks, dim=0)
        flow_s = torch.cat(flow_s_chunks, dim=0)
        flow_d = torch.cat(flow_d_chunks, dim=0)
        edge_prob = torch.cat(edge_prob_chunks, dim=0)
        alpha_dyn = torch.cat(alpha_dyn_chunks, dim=0)
        g1_all = torch.cat(g1_chunks, dim=0)
        edge_logit = torch.cat(edge_logit_chunks, dim=0)

        return {
            "flow_pred": self._format_edge_output(flow_final),
            "flow_cond": self._format_edge_output(flow_cond),
            "flow_static": self._format_edge_output(flow_s),
            "flow_dyn": self._format_edge_output(flow_d),
            "edge_prob": self._format_edge_output(edge_prob),
            "alpha_sd": self._format_edge_output(alpha_dyn),
            "g1": self._format_edge_output(g1_all),
            "od_edge_logits": self._format_edge_output(edge_logit),
            "onset_logit": None,
            "burst_logit": None,
            "od_node_logits": node_logit,
        }


ODDecoder = MultiTaskGatedDecoder
Decoder = MultiTaskGatedDecoder
EdgeDecoder = MultiTaskGatedDecoder
TwoHeadDecoder = MultiTaskGatedDecoder