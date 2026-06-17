# model/coblock.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class EdgeNodeCoBlock(nn.Module):
    """
    Edge-node cooperative update block with separated outgoing/incoming aggregation.

    Inputs:
        node_x: [N, H]
        edge_x: [E, H]
        edge_index: [2, E], where edge_index[0] is src and edge_index[1] is dst

    Outputs:
        node_x_new: [N, H]
        edge_x_new: [E, H]
    """

    def __init__(
        self,
        hidden_dim: int,
        channels: int = 4,
        dropout: float = 0.1,
        use_directional_co: int = 0,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.channels = max(1, int(channels))
        self.drop = nn.Dropout(float(dropout))
        self.use_directional_co = bool(int(use_directional_co))

        self.edge_filters_out = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                for _ in range(self.channels)
            ]
        )
        self.edge_filters_in = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                for _ in range(self.channels)
            ]
        )

        self.norm_node = nn.LayerNorm(self.hidden_dim)
        self.norm_agg_out = nn.LayerNorm(self.hidden_dim)
        self.norm_agg_in = nn.LayerNorm(self.hidden_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.norm_edge = nn.LayerNorm(self.hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        if self.use_directional_co:
            self.node_out_role = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.node_in_role = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.norm_out_role = nn.LayerNorm(self.hidden_dim)
            self.norm_in_role = nn.LayerNorm(self.hidden_dim)
        else:
            self.node_out_role = None
            self.node_in_role = None
            self.norm_out_role = None
            self.norm_in_role = None

        self.edge_chunk_size = 1024

    def _edge_to_node(
        self,
        node_x: torch.Tensor,
        edge_x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        src, dst = edge_index

        device = node_x.device
        E = edge_x.size(0)
        H = edge_x.size(1)
        N = int(num_nodes)

        agg_out = torch.zeros(N, H, device=device, dtype=edge_x.dtype)
        agg_in = torch.zeros(N, H, device=device, dtype=edge_x.dtype)

        deg_out = torch.zeros(N, 1, device=device, dtype=edge_x.dtype)
        deg_in = torch.zeros(N, 1, device=device, dtype=edge_x.dtype)

        ones = torch.ones(E, 1, device=device, dtype=edge_x.dtype)

        for filt_out, filt_in in zip(self.edge_filters_out, self.edge_filters_in):
            ex_out = filt_out(edge_x)
            ex_in = filt_in(edge_x)

            agg_out.index_add_(0, src, ex_out)
            agg_in.index_add_(0, dst, ex_in)

        deg_out.index_add_(0, src, ones)
        deg_in.index_add_(0, dst, ones)

        agg_out = agg_out / deg_out.clamp_min(1.0)
        agg_in = agg_in / deg_in.clamp_min(1.0)

        agg_out = self.norm_agg_out(agg_out)
        agg_in = self.norm_agg_in(agg_in)

        node_input = torch.cat(
            [
                self.norm_node(node_x),
                agg_out,
                agg_in,
            ],
            dim=-1,
        )

        node_delta = self.node_mlp(node_input)
        return node_x + node_delta

    def _node_to_edge_chunked(
        self,
        node_x: torch.Tensor,
        edge_x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index

        device = edge_x.device
        E = edge_x.size(0)
        H = edge_x.size(1)

        edge_norm = self.norm_edge(edge_x)
        edge_delta = torch.empty(E, H, device=device, dtype=edge_x.dtype)

        for start in range(0, E, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, E)

            src_chunk = src[start:end]
            dst_chunk = dst[start:end]

            u_src = node_x.index_select(0, src_chunk)
            u_dst = node_x.index_select(0, dst_chunk)
            e_chunk = edge_norm[start:end]

            if self.use_directional_co:
                src_role = self.node_out_role(u_src)
                dst_role = self.node_in_role(u_dst)

                src_role = torch.tanh(self.norm_out_role(src_role))
                dst_role = torch.tanh(self.norm_in_role(dst_role))

                interaction = torch.abs(src_role - dst_role)

                edge_input = torch.cat(
                    [
                        e_chunk,
                        src_role,
                        dst_role,
                        interaction,
                    ],
                    dim=-1,
                )
            else:
                diff = torch.abs(u_src - u_dst)

                edge_input = torch.cat(
                    [
                        e_chunk,
                        u_src,
                        u_dst,
                        diff,
                    ],
                    dim=-1,
                )

            edge_delta[start:end] = self.edge_mlp(edge_input)

        return edge_x + edge_delta

    def forward(
        self,
        node_x: torch.Tensor,
        edge_x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int = None,
    ):
        if num_nodes is None:
            if edge_index.numel() == 0:
                num_nodes = int(node_x.size(0))
            else:
                src, dst = edge_index
                num_nodes = int(max(int(src.max()), int(dst.max())) + 1)

        edge_msg = self.drop(edge_x)

        node_x_new = self._edge_to_node(
            node_x=node_x,
            edge_x=edge_msg,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )

        edge_x_new = self._node_to_edge_chunked(
            node_x=node_x_new,
            edge_x=edge_x,
            edge_index=edge_index,
        )

        return node_x_new, edge_x_new

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"channels={self.channels}, "
            f"use_directional_co={int(self.use_directional_co)}, "
            f"edge_chunk_size={self.edge_chunk_size}"
        )