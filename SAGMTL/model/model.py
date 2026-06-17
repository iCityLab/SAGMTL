# model/model.py
# -*- coding: utf-8 -*-
import importlib
import inspect
from typing import Dict, Any

import torch
import torch.nn as nn

from .encoder import TimeEdgeEncoder


def _find_decoder_class():
    dec_mod = importlib.import_module(".decoder", __package__)
    dec_mod = importlib.reload(dec_mod)

    candidates = [
        "MultiTaskGatedDecoder",
        "ODDecoder",
        "Decoder",
        "EdgeDecoder",
        "TwoHeadDecoder",
    ]

    for name in candidates:
        if hasattr(dec_mod, name):
            return getattr(dec_mod, name), name

    raise ImportError("No decoder class found in model/decoder.py.")


def _safe_instantiate(cls, **kwargs):
    sig = inspect.signature(cls.__init__)
    ok = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**ok)


class ODModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hidden_dim = int(getattr(args, "hidden_dim", 96))
        self.edge_dim = int(getattr(args, "edge_dim", 128))
        self.time_dim = int(getattr(args, "time_dim", 64))
        self.horizon = int(getattr(args, "horizon", 6))

        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")

        self.edge_chunk_size = int(getattr(args, "edge_chunk_size", 1024))

        # Encoder settings
        self.enc_dropout = float(getattr(args, "enc_dropout", 0.1))

        self.use_edge_history = bool(int(getattr(args, "use_edge_history", 1)))
        self.edge_history_input = str(
            getattr(args, "edge_history_input", "flow_exist")
        ).lower().strip()
        self.edge_history_dim = int(getattr(args, "edge_history_dim", self.hidden_dim))
        self.edge_history_chunk_size = int(
            getattr(
                args,
                "edge_history_chunk_size",
                getattr(args, "edge_chunk_size", 4096),
            )
        )
        self.edge_history_fuse = str(
            getattr(args, "edge_history_fuse", "concat")
        ).lower().strip()

        self.num_co_layers = int(getattr(args, "num_co_layers", 2))
        self.co_channels = int(getattr(args, "co_channels", 4))
        self.co_dropout = float(getattr(args, "co_dropout", 0.1))
        self.use_directional_co = bool(int(getattr(args, "use_directional_co", 0)))

        self.use_time_dyn = bool(int(getattr(args, "use_time_dyn", 1)))
        self.use_static_node = bool(int(getattr(args, "use_static_node", 1)))
        self.static_weight = float(getattr(args, "static_weight", 0.5))
        self.dyn_weight = float(getattr(args, "dyn_weight", 0.5))
        self.normalize_blend = bool(int(getattr(args, "normalize_blend", 1)))

        self.static_dyn_fuse_mode = str(
            getattr(args, "static_dyn_fuse_mode", "concat")
        ).lower().strip()

        self.use_time_attention = bool(int(getattr(args, "use_time_attention", 0)))
        self.time_attn_heads = int(getattr(args, "time_attn_heads", 4))
        self.time_attn_layers = int(getattr(args, "time_attn_layers", 1))
        self.time_attn_ff_mult = int(getattr(args, "time_attn_ff_mult", 2))
        self.time_attn_causal = bool(int(getattr(args, "time_attn_causal", 0)))
        self.time_pool_mode = str(getattr(args, "time_pool_mode", "last")).lower().strip()

        self.use_spatial_attn = bool(int(getattr(args, "use_spatial_attn", 1)))
        self.spatial_dist_path = str(getattr(args, "spatial_dist_path", ""))
        self.spatial_attn_heads = int(getattr(args, "spatial_attn_heads", 4))
        self.spatial_attn_dropout = float(
            getattr(args, "spatial_attn_dropout", self.enc_dropout)
        )
        self.spatial_attn_res_weight = float(
            getattr(args, "spatial_attn_res_weight", 0.15)
        )
        self.spatial_attn_exclude_self = bool(
            int(getattr(args, "spatial_attn_exclude_self", 1))
        )
        self.use_spatial_dist_decay = bool(
            int(getattr(args, "use_spatial_dist_decay", 1))
        )
        self.spatial_dist_decay_init = float(
            getattr(args, "spatial_dist_decay_init", 0.1)
        )
        self.spatial_dist_decay_per_head = bool(
            int(getattr(args, "spatial_dist_decay_per_head", 0))
        )
        self.spatial_dist_norm = str(
            getattr(args, "spatial_dist_norm", "max")
        ).lower().strip()

        # Decoder settings
        self.dec_dropout = float(getattr(args, "dec_dropout", 0.1))
        self.role_dim = int(getattr(args, "role_dim", self.hidden_dim))

        self.use_gate1 = bool(int(getattr(args, "use_gate1", 1)))
        self.use_last_flow_gate = bool(int(getattr(args, "use_last_flow_gate", 1)))
        self.use_sd_gate = bool(int(getattr(args, "use_sd_gate", 1)))
        self.fixed_alpha = float(getattr(args, "fixed_alpha", 0.7))
        self.use_od_gate = bool(int(getattr(args, "use_od_gate", 1)))
        self.gate2_detach = bool(int(getattr(args, "gate2_detach", 1)))

        self.use_residual_bias = bool(int(getattr(args, "use_residual_bias", 1)))
        self.use_single_head_fusion = bool(
            int(getattr(args, "use_single_head_fusion", 0))
        )

        self.dyn_decoder_mode = str(
            getattr(args, "dyn_decoder_mode", "residual")
        ).lower().strip()
        self.dyn_res_use_tanh = bool(int(getattr(args, "dyn_res_use_tanh", 1)))
        self.dyn_res_scale = float(getattr(args, "dyn_res_scale", 1.0))
        self.max_log_gain = float(getattr(args, "max_log_gain", 2.5))

        self.use_edge_history_in_decoder = bool(
            int(getattr(args, "use_edge_history_in_decoder", 1))
        )
        self.edge_history_decoder_detach = bool(
            int(getattr(args, "edge_history_decoder_detach", 0))
        )
        self.static_branch_mode = str(
            getattr(args, "static_branch_mode", "static_roles")
        ).lower().strip()
        self.sd_fusion_mode = str(
            getattr(args, "sd_fusion_mode", "adaptive")
        ).lower().strip()

        self.use_static_basis_head = bool(
            int(getattr(args, "use_static_basis_head", 0))
        )
        self.static_basis_rank = int(getattr(args, "static_basis_rank", 4))
        self.static_basis_dropout = float(
            getattr(args, "static_basis_dropout", self.dec_dropout)
        )

        self.use_dyn_query_head = bool(int(getattr(args, "use_dyn_query_head", 0)))
        self.dyn_query_heads = int(getattr(args, "dyn_query_heads", 4))
        self.dyn_query_ff_mult = int(getattr(args, "dyn_query_ff_mult", 2))
        self.dyn_query_dropout = float(
            getattr(args, "dyn_query_dropout", self.dec_dropout)
        )
        self.dyn_query_tokens = int(getattr(args, "dyn_query_tokens", 4))

        self.dyn_share_floor = float(getattr(args, "dyn_share_floor", 0.35))
        self.static_scale = float(getattr(args, "static_scale", 0.60))

        self.encoder = _safe_instantiate(
            TimeEdgeEncoder,
            hidden_dim=self.hidden_dim,
            edge_dim=self.edge_dim,
            time_dim=self.time_dim,
            dropout=self.enc_dropout,
            enc_dropout=self.enc_dropout,
            edge_chunk_size=self.edge_chunk_size,

            use_edge_history=int(self.use_edge_history),
            edge_history_input=self.edge_history_input,
            edge_history_dim=self.edge_history_dim,
            edge_history_chunk_size=self.edge_history_chunk_size,
            edge_history_fuse=self.edge_history_fuse,

            num_co_layers=self.num_co_layers,
            co_channels=self.co_channels,
            co_dropout=self.co_dropout,
            use_directional_co=int(self.use_directional_co),

            use_time_dyn=int(self.use_time_dyn),
            use_static_node=int(self.use_static_node),
            static_weight=self.static_weight,
            dyn_weight=self.dyn_weight,
            normalize_blend=int(self.normalize_blend),
            static_dyn_fuse_mode=self.static_dyn_fuse_mode,

            use_time_attention=int(self.use_time_attention),
            time_attn_heads=self.time_attn_heads,
            time_attn_layers=self.time_attn_layers,
            time_attn_ff_mult=self.time_attn_ff_mult,
            time_attn_causal=int(self.time_attn_causal),
            time_pool_mode=self.time_pool_mode,

            use_spatial_attn=int(self.use_spatial_attn),
            spatial_dist_path=self.spatial_dist_path,
            spatial_attn_heads=self.spatial_attn_heads,
            spatial_attn_dropout=self.spatial_attn_dropout,
            spatial_attn_res_weight=self.spatial_attn_res_weight,
            spatial_attn_exclude_self=int(self.spatial_attn_exclude_self),
            use_spatial_dist_decay=int(self.use_spatial_dist_decay),
            spatial_dist_decay_init=self.spatial_dist_decay_init,
            spatial_dist_decay_per_head=int(self.spatial_dist_decay_per_head),
            spatial_dist_norm=self.spatial_dist_norm,
        )

        DecoderCls, decoder_name = _find_decoder_class()
        self.decoder = _safe_instantiate(
            DecoderCls,
            hidden_dim=self.hidden_dim,
            edge_dim=self.edge_dim,
            horizon=self.horizon,
            role_dim=self.role_dim,
            dropout=self.dec_dropout,
            edge_chunk_size=self.edge_chunk_size,

            use_gate1=int(self.use_gate1),
            use_last_flow_gate=int(self.use_last_flow_gate),
            use_sd_gate=int(self.use_sd_gate),
            fixed_alpha=self.fixed_alpha,
            use_od_gate=int(self.use_od_gate),
            gate2_detach=int(self.gate2_detach),

            use_residual_bias=int(self.use_residual_bias),
            use_single_head_fusion=int(self.use_single_head_fusion),

            dyn_decoder_mode=self.dyn_decoder_mode,
            dyn_res_use_tanh=int(self.dyn_res_use_tanh),
            dyn_res_scale=self.dyn_res_scale,
            max_log_gain=self.max_log_gain,

            use_edge_history_in_decoder=int(self.use_edge_history_in_decoder),
            edge_history_decoder_detach=int(self.edge_history_decoder_detach),
            static_branch_mode=self.static_branch_mode,
            sd_fusion_mode=self.sd_fusion_mode,

            use_static_basis_head=int(self.use_static_basis_head),
            static_basis_rank=self.static_basis_rank,
            static_basis_dropout=self.static_basis_dropout,

            use_dyn_query_head=int(self.use_dyn_query_head),
            dyn_query_heads=self.dyn_query_heads,
            dyn_query_ff_mult=self.dyn_query_ff_mult,
            dyn_query_dropout=self.dyn_query_dropout,
            dyn_query_tokens=self.dyn_query_tokens,

            dyn_share_floor=self.dyn_share_floor,
            static_scale=self.static_scale,
        )

        self._decoder_class_name = decoder_name

        self.mse = nn.MSELoss(reduction="mean")
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        enc_out = self.encoder(batch)
        dec_out = self.decoder(enc_out, batch)

        return {
            "flow_pred": dec_out.get("flow_pred", None),
            "flow_cond": dec_out.get("flow_cond", None),
            "flow_static": dec_out.get("flow_static", None),
            "flow_dyn": dec_out.get("flow_dyn", None),

            "edge_prob": dec_out.get("edge_prob", None),
            "alpha_sd": dec_out.get("alpha_sd", None),
            "g1": dec_out.get("g1", None),

            "od_node_logits": dec_out.get("od_node_logits", None),
            "od_edge_logits": dec_out.get("od_edge_logits", None),

            "onset_logit": dec_out.get("onset_logit", None),
            "burst_logit": dec_out.get("burst_logit", None),
        }

    @torch.no_grad()
    def infer_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.eval()
        ret = self.forward(batch)

        if ret.get("edge_prob") is not None:
            ret["od_edge_prob"] = ret["edge_prob"]
        elif ret.get("od_edge_logits") is not None:
            ret["od_edge_prob"] = torch.sigmoid(ret["od_edge_logits"])

        if ret.get("od_node_logits") is not None:
            ret["od_node_prob"] = torch.sigmoid(ret["od_node_logits"])

        if ret.get("onset_logit") is not None:
            ret["onset_prob"] = torch.sigmoid(ret["onset_logit"])

        if ret.get("burst_logit") is not None:
            ret["burst_prob"] = torch.sigmoid(ret["burst_logit"])

        return ret