# utils/loss_utils.py
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Shape helpers
# -----------------------------------------------------------------------------

def ensure_edge_seq(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    raise ValueError(f"edge tensor must be 1D or 2D, got shape={tuple(x.shape)}")


def ensure_node_seq(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"node tensor must be 2D or 3D, got shape={tuple(x.shape)}")


def get_flow_target_seq(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "edge_labels_seq" in batch:
        return ensure_edge_seq(batch["edge_labels_seq"].float())
    return ensure_edge_seq(batch["edge_labels"].float())


def get_edge_exist_seq(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "y_od_seq" in batch:
        return ensure_edge_seq(batch["y_od_seq"].float())
    if "y_od" in batch:
        return ensure_edge_seq(batch["y_od"].float())
    y_flow = get_flow_target_seq(batch)
    return (y_flow > 0.0).float()


def get_node_label_seq(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "node_od_label_seq" in batch:
        return ensure_node_seq(batch["node_od_label_seq"].float())
    return ensure_node_seq(batch["node_od_label"].float())


def _zero_like_ref(*xs: Optional[torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
    for x in xs:
        if x is not None and torch.is_tensor(x):
            return torch.tensor(0.0, device=x.device)
    if device is not None:
        return torch.tensor(0.0, device=device)
    return torch.tensor(0.0)


# -----------------------------------------------------------------------------
# Structure supervision
# -----------------------------------------------------------------------------

def dynamic_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    y = labels.float().to(logits.device)
    x = logits.float()
    if y.dim() != 2 or x.dim() != 2:
        raise ValueError(f"dynamic_bce_loss expects 2D tensors, got logits={x.shape}, labels={y.shape}")

    n = y.shape[0]
    pos = y.sum(dim=0).clamp(min=1.0)
    neg = (n - y.sum(dim=0)).clamp(min=1.0)
    pos_weight = (neg / pos).to(x.device)
    return F.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight, reduction="mean")


def focal_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    y = labels.float().to(logits.device)
    x = logits.float()

    bce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
    p = torch.sigmoid(x)
    pt = y * p + (1.0 - y) * (1.0 - p)
    alpha_t = y * float(alpha) + (1.0 - y) * (1.0 - float(alpha))
    loss = alpha_t * ((1.0 - pt).clamp_min(1e-6) ** float(gamma)) * bce

    if sample_weight is not None:
        loss = loss * sample_weight.to(loss.device, dtype=loss.dtype)
    return loss.mean()


def build_transition_weight(
    y_cur: torch.Tensor,
    y_prev: Optional[torch.Tensor] = None,
    transition_boost: float = 2.0,
) -> torch.Tensor:
    y_cur_bin = (y_cur > 0.5).float()
    if y_prev is None:
        return torch.ones_like(y_cur_bin)
    y_prev_bin = (y_prev > 0.5).float().to(y_cur_bin.device)
    trans = (y_cur_bin != y_prev_bin).float()
    return 1.0 + float(transition_boost) * trans


def edge_activation_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str = "focal",
    alpha: float = 0.75,
    gamma: float = 2.0,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss_type = str(loss_type or "focal").lower().strip()
    y = labels.float().to(logits.device)
    x = logits.float()

    if loss_type == "focal":
        return focal_bce_with_logits(
            x,
            y,
            alpha=alpha,
            gamma=gamma,
            sample_weight=sample_weight,
        )

    if loss_type == "bce":
        loss = F.binary_cross_entropy_with_logits(x, y, reduction="none")
        if sample_weight is not None:
            loss = loss * sample_weight.to(loss.device, dtype=loss.dtype)
        return loss.mean()

    raise ValueError(f"Unsupported edge activation loss type: {loss_type}")


# Backward-compatible name.
edge_exist_loss_with_logits = edge_activation_loss_with_logits


def sample_edge_indices_for_bce(
    edge_index: torch.Tensor,
    y_exist: torch.Tensor,
    last_flow: Optional[torch.Tensor],
    num_nodes: int,
    neg_ratio: float = 1.0,
    neg_active_only: bool = True,
) -> torch.Tensor:
    y = y_exist.view(-1)
    pos_idx = torch.nonzero(y > 0.5, as_tuple=False).view(-1)
    neg_idx = torch.nonzero(y <= 0.5, as_tuple=False).view(-1)

    if pos_idx.numel() == 0 or float(neg_ratio) <= 0.0 or neg_idx.numel() == 0:
        return pos_idx

    if bool(neg_active_only):
        active = torch.zeros((int(num_nodes),), dtype=torch.bool, device=y.device)

        src_p = edge_index[0, pos_idx]
        dst_p = edge_index[1, pos_idx]
        active[src_p] = True
        active[dst_p] = True

        if last_flow is not None and last_flow.numel() == y.numel():
            lf = last_flow.view(-1).to(y.device)
            active_edges = torch.nonzero(lf > 0, as_tuple=False).view(-1)
            if active_edges.numel() > 0:
                src_a = edge_index[0, active_edges]
                dst_a = edge_index[1, active_edges]
                active[src_a] = True
                active[dst_a] = True

        src_n = edge_index[0, neg_idx]
        dst_n = edge_index[1, neg_idx]
        keep = active[src_n] & active[dst_n]
        neg_idx = neg_idx[keep]

        if neg_idx.numel() == 0:
            return pos_idx

    k = int(pos_idx.numel() * float(neg_ratio))
    k = min(k, neg_idx.numel())
    if k <= 0:
        return pos_idx

    perm = torch.randperm(neg_idx.numel(), device=y.device)[:k]
    neg_sample = neg_idx[perm]
    return torch.cat([pos_idx, neg_sample], dim=0)


# -----------------------------------------------------------------------------
# Base flow loss with intensity compensation
# -----------------------------------------------------------------------------

def _intensity_compensation_weight(
    target_pos: torch.Tensor,
    mode: str = "legacy",
    lambda_comp: float = 1.0,
) -> torch.Tensor:
    mode = str(mode or "legacy").lower().strip()

    if target_pos.numel() == 0:
        return torch.ones_like(target_pos)

    if mode == "none":
        return torch.ones_like(target_pos)

    if mode == "legacy":
        scale = torch.log1p(torch.clamp(target_pos, min=0.0))
        scale = scale / (scale.mean().detach() + 1e-6)
        return 1.0 + float(lambda_comp) * scale

    if mode == "direct_log":
        scale = torch.log1p(torch.clamp(target_pos, min=0.0))
        denom = scale.mean().detach()
        if (not torch.isfinite(denom)) or float(denom.item()) <= 0.0:
            return torch.ones_like(target_pos)
        return scale / denom

    if mode == "direct_raw":
        scale = torch.clamp(target_pos, min=0.0)
        denom = scale.mean().detach()
        if (not torch.isfinite(denom)) or float(denom.item()) <= 0.0:
            return torch.ones_like(target_pos)
        return scale / denom

    raise ValueError(f"Unsupported compensation weight mode: {mode}")


def compensated_positive_flow_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    flow_loss_space: str = "log1p",
    flow_loss_type: str = "smoothl1",
    comp_weight_mode: str = "legacy",
    lambda_comp: float = 1.0,
) -> torch.Tensor:
    if pred is None or target is None:
        return _zero_like_ref(pred, target)

    pred = pred.reshape(-1)
    target = target.reshape(-1).to(pred.device)

    pos_mask = target > 0.0
    if not pos_mask.any():
        return torch.tensor(0.0, device=pred.device)

    pred_pos = torch.clamp(pred[pos_mask], min=0.0)
    target_pos = torch.clamp(target[pos_mask], min=0.0)

    loss_space = str(flow_loss_space or "log1p").lower().strip()
    loss_type = str(flow_loss_type or "smoothl1").lower().strip()

    if loss_space == "log1p":
        pred_in = torch.log1p(pred_pos)
        target_in = torch.log1p(target_pos)
    elif loss_space == "raw":
        pred_in = pred_pos
        target_in = target_pos
    else:
        raise ValueError(f"Unsupported flow_loss_space: {flow_loss_space}")

    if loss_type == "smoothl1":
        base = F.smooth_l1_loss(pred_in, target_in, reduction="none")
    elif loss_type == "mse":
        base = (pred_in - target_in) ** 2
    else:
        raise ValueError(f"Unsupported flow_loss_type: {flow_loss_type}")

    weight = _intensity_compensation_weight(
        target_pos=target_pos,
        mode=comp_weight_mode,
        lambda_comp=lambda_comp,
    ).to(base.device, dtype=base.dtype)

    return (base * weight).mean()


# Backward-compatible wrappers.
def positive_flow_reg_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    flow_loss_space: str = "log1p",
    flow_loss_type: str = "smoothl1",
) -> torch.Tensor:
    return compensated_positive_flow_loss(
        pred=pred,
        target=target,
        flow_loss_space=flow_loss_space,
        flow_loss_type=flow_loss_type,
        comp_weight_mode="none",
        lambda_comp=0.0,
    )


def weighted_positive_flow_reg_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    flow_loss_space: str = "log1p",
    flow_loss_type: str = "smoothl1",
    peak_weight_mode: str = "legacy",
    peak_alpha: float = 1.0,
) -> torch.Tensor:
    return compensated_positive_flow_loss(
        pred=pred,
        target=target,
        flow_loss_space=flow_loss_space,
        flow_loss_type=flow_loss_type,
        comp_weight_mode=peak_weight_mode,
        lambda_comp=peak_alpha,
    )


def positive_log_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return positive_flow_reg_loss(
        pred,
        target,
        flow_loss_space="log1p",
        flow_loss_type="smoothl1",
    )


def weighted_positive_log_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    peak_alpha: float = 1.0,
) -> torch.Tensor:
    return compensated_positive_flow_loss(
        pred,
        target,
        flow_loss_space="log1p",
        flow_loss_type="smoothl1",
        comp_weight_mode="legacy",
        lambda_comp=peak_alpha,
    )


# -----------------------------------------------------------------------------
# Flow regularizers
# -----------------------------------------------------------------------------

def hard_negative_zero_loss(
    pred_seq: torch.Tensor,
    target_seq: torch.Tensor,
    top_ratio: float = 0.01,
    min_topk: int = 64,
    max_topk: int = 2048,
) -> torch.Tensor:
    if pred_seq is None or target_seq is None:
        return _zero_like_ref(pred_seq, target_seq)

    pred_seq = ensure_edge_seq(pred_seq)
    target_seq = ensure_edge_seq(target_seq)

    h_max = min(pred_seq.shape[0], target_seq.shape[0])
    if h_max <= 0:
        return torch.tensor(0.0, device=pred_seq.device)

    losses = []
    for h in range(h_max):
        pred_h = pred_seq[h].reshape(-1)
        true_h = target_seq[h].reshape(-1).to(pred_h.device)
        zero_mask = true_h <= 0.0
        if not zero_mask.any():
            continue

        pred_zero = torch.clamp(pred_h[zero_mask], min=0.0)
        n_zero = int(pred_zero.numel())
        if n_zero <= 0:
            continue

        k = max(int(n_zero * float(top_ratio)), int(min_topk))
        k = min(k, n_zero, int(max_topk))
        if k <= 0:
            continue

        topv, _ = torch.topk(pred_zero, k=k, largest=True, sorted=False)
        losses.append(F.smooth_l1_loss(topv, torch.zeros_like(topv), reduction="mean"))

    if not losses:
        return torch.tensor(0.0, device=pred_seq.device)
    return torch.stack(losses).mean()


def volatility_smooth_l1_loss(
    pred_seq: torch.Tensor,
    target_seq: torch.Tensor,
    last_flow: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred_seq is None or target_seq is None:
        return _zero_like_ref(pred_seq, target_seq)

    pred_seq = ensure_edge_seq(pred_seq)
    target_seq = ensure_edge_seq(target_seq)

    h_max = min(pred_seq.shape[0], target_seq.shape[0])
    if h_max <= 0:
        return torch.tensor(0.0, device=pred_seq.device)

    pred_seq = pred_seq[:h_max]
    target_seq = target_seq[:h_max].to(pred_seq.device)

    edge_count = target_seq.shape[-1]
    if last_flow is None:
        last_flow_vec = torch.zeros((edge_count,), device=pred_seq.device, dtype=pred_seq.dtype)
    else:
        if not torch.is_tensor(last_flow):
            last_flow_vec = torch.as_tensor(last_flow, device=pred_seq.device, dtype=pred_seq.dtype)
        else:
            last_flow_vec = last_flow.to(pred_seq.device, dtype=pred_seq.dtype)
        last_flow_vec = last_flow_vec.view(-1)
        if last_flow_vec.numel() != edge_count:
            last_flow_vec = torch.zeros((edge_count,), device=pred_seq.device, dtype=pred_seq.dtype)

    prev_pred = torch.cat([last_flow_vec.unsqueeze(0), pred_seq[:-1]], dim=0)
    prev_true = torch.cat([last_flow_vec.unsqueeze(0), target_seq[:-1]], dim=0)

    delta_pred = pred_seq - prev_pred
    delta_true = target_seq - prev_true

    active_mask = (torch.abs(delta_true) > eps) | (target_seq > eps) | (prev_true > eps)
    if not active_mask.any():
        return torch.tensor(0.0, device=pred_seq.device)

    return F.smooth_l1_loss(delta_pred[active_mask], delta_true[active_mask], reduction="mean")


def od_conservation_loss(
    pred_flow: torch.Tensor,
    true_flow: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    num_nodes: int,
    out_weight: float = 1.0,
    in_weight: float = 1.0,
) -> torch.Tensor:
    if pred_flow is None or true_flow is None or pred_flow.numel() == 0:
        return _zero_like_ref(pred_flow, true_flow)

    pred_flow = ensure_edge_seq(pred_flow)
    true_flow = ensure_edge_seq(true_flow).to(pred_flow.device)

    h_max = min(pred_flow.shape[0], true_flow.shape[0])
    pred_flow = pred_flow[:h_max]
    true_flow = true_flow[:h_max]

    H, E = pred_flow.shape
    if E == 0:
        return torch.tensor(0.0, device=pred_flow.device)

    src_idx = edge_src.long().to(pred_flow.device).view(1, -1).expand(H, -1)
    dst_idx = edge_dst.long().to(pred_flow.device).view(1, -1).expand(H, -1)

    pred_out = torch.zeros((H, int(num_nodes)), device=pred_flow.device, dtype=pred_flow.dtype)
    true_out = torch.zeros((H, int(num_nodes)), device=pred_flow.device, dtype=true_flow.dtype)
    pred_in = torch.zeros((H, int(num_nodes)), device=pred_flow.device, dtype=pred_flow.dtype)
    true_in = torch.zeros((H, int(num_nodes)), device=pred_flow.device, dtype=true_flow.dtype)

    pred_out.scatter_add_(1, src_idx, pred_flow)
    true_out.scatter_add_(1, src_idx, true_flow)
    pred_in.scatter_add_(1, dst_idx, pred_flow)
    true_in.scatter_add_(1, dst_idx, true_flow)

    total_w = float(out_weight) + float(in_weight)
    if total_w <= 0.0:
        return torch.tensor(0.0, device=pred_flow.device)

    loss_out = (
        F.smooth_l1_loss(pred_out, true_out, reduction="mean")
        if float(out_weight) > 0.0
        else torch.tensor(0.0, device=pred_flow.device)
    )
    loss_in = (
        F.smooth_l1_loss(pred_in, true_in, reduction="mean")
        if float(in_weight) > 0.0
        else torch.tensor(0.0, device=pred_flow.device)
    )
    return (float(out_weight) * loss_out + float(in_weight) * loss_in) / (total_w + 1e-8)


# -----------------------------------------------------------------------------
# Scheduling and full objective
# -----------------------------------------------------------------------------

def scheduled_scale(epoch: int, warmup_epochs: int) -> float:
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs > 0 and int(epoch) <= warmup_epochs:
        return float(epoch) / float(warmup_epochs)
    return 1.0


def _resolve_base_split(args: Any) -> Tuple[float, float, float]:
    if hasattr(args, "lambda_base"):
        lambda_base = float(getattr(args, "lambda_base"))
        lambda_cond_ref = float(getattr(args, "lambda_base_cond", 0.5 * lambda_base))
        lambda_final_ref = float(getattr(args, "lambda_base_final", lambda_base - lambda_cond_ref))
    else:
        lambda_cond_ref = float(getattr(args, "reg_lambda_cond", 1.5))
        lambda_final_ref = float(getattr(args, "reg_lambda_final", 1.5))
        lambda_base = lambda_cond_ref + lambda_final_ref
    return lambda_base, lambda_cond_ref, lambda_final_ref


def scheduled_base_weight(epoch: int, args: Any) -> Dict[str, float]:
    lambda_base, lambda_cond_ref, lambda_final_ref = _resolve_base_split(args)

    structure_pretrain_epochs = int(getattr(args, "structure_pretrain_epochs", 5))
    reg_warmup_epochs = int(getattr(args, "reg_warmup_epochs", max(structure_pretrain_epochs, 8)))
    structure_pretrain_base_scale = float(getattr(args, "structure_pretrain_base_scale", getattr(args, "structure_pretrain_cond_scale", 0.30)))

    warm_epochs = max(reg_warmup_epochs, structure_pretrain_epochs)
    epoch = int(epoch)

    if epoch <= structure_pretrain_epochs:
        cur_cond = lambda_cond_ref * structure_pretrain_base_scale
        cur_final = 0.0
    elif epoch <= warm_epochs:
        denom = max(1, warm_epochs - structure_pretrain_epochs)
        p = float(epoch - structure_pretrain_epochs) / float(denom)
        cur_cond = lambda_cond_ref * (structure_pretrain_base_scale + (1.0 - structure_pretrain_base_scale) * p)
        cur_final = lambda_final_ref * p
    else:
        cur_cond = lambda_cond_ref
        cur_final = lambda_final_ref

    return {
        "lambda_base": float(lambda_base),
        "lambda_base_current": float(cur_cond + cur_final),
        "lambda_base_cond_part": float(cur_cond),
        "lambda_base_final_part": float(cur_final),
    }


def log_loss_config(logger: Any, args: Any) -> None:
    schedule = scheduled_base_weight(epoch=10**9, args=args)
    lambda_comp = float(getattr(args, "lambda_comp", getattr(args, "peak_alpha", 1.0)))
    comp_weight_mode = str(getattr(args, "comp_weight_mode", getattr(args, "peak_weight_mode", "legacy"))).lower().strip()
    flow_loss_space = str(getattr(args, "flow_loss_space", "log1p")).lower().strip()
    flow_loss_type = str(getattr(args, "flow_loss_type", "smoothl1")).lower().strip()

    w_base = float(getattr(args, "w_base", getattr(args, "w_flow", 1.0)))
    w_zero = float(getattr(args, "w_zero", 0.035))
    w_vol = float(getattr(args, "w_vol", 0.1))
    w_cons = float(getattr(args, "w_cons", 0.1))
    w_node = float(getattr(args, "w_node", getattr(args, "w_od", 0.08)))
    w_edge = float(getattr(args, "w_edge", 0.08))

    logger.info(
        "[loss] total = "
        "w_base * lambda_base(t) * L_base "
        "+ w_zero * L_zero "
        "+ w_vol * L_vol "
        "+ cons_scale(t) * w_cons * L_cons "
        "+ aux_scale(t) * (w_node * L_node + w_edge * L_edge)"
    )
    logger.info(
        "[loss-detail] "
        f"lambda_base={schedule['lambda_base']:.3f} "
        f"lambda_comp={lambda_comp:.3f} "
        f"w_base={w_base:.3f} w_zero={w_zero:.3f} w_vol={w_vol:.3f} "
        f"w_cons={w_cons:.3f} w_node={w_node:.3f} w_edge={w_edge:.3f} | "
        f"flow_loss_space={flow_loss_space} flow_loss_type={flow_loss_type} comp_weight_mode={comp_weight_mode} | "
        f"structure_pretrain_epochs={int(getattr(args, 'structure_pretrain_epochs', 5))} "
        f"reg_warmup_epochs={int(getattr(args, 'reg_warmup_epochs', 8))} "
        f"structure_pretrain_base_scale={float(getattr(args, 'structure_pretrain_base_scale', getattr(args, 'structure_pretrain_cond_scale', 0.30))):.3f} | "
        f"aux_warmup_epochs={int(getattr(args, 'aux_warmup_epochs', 15))} "
        f"cons_warmup_epochs={int(getattr(args, 'cons_warmup_epochs', 10))} | "
        f"edge_loss_type={str(getattr(args, 'edge_loss_type', getattr(args, 'exist_loss_type', 'focal'))).lower().strip()} "
        f"edge_alpha={float(getattr(args, 'edge_alpha', getattr(args, 'exist_alpha', 0.75))):.3f} "
        f"edge_gamma={float(getattr(args, 'edge_gamma', getattr(args, 'exist_gamma', 2.0))):.3f} "
        f"transition_boost={float(getattr(args, 'transition_boost', 2.0)):.3f} "
        f"use_transition_weight={int(getattr(args, 'use_transition_weight', 1))}"
    )


def compute_train_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args: Any,
    epoch: int,
) -> Dict[str, torch.Tensor]:
    y_flow_seq = get_flow_target_seq(batch)
    y_exist_seq = get_edge_exist_seq(batch)
    y_node_seq = get_node_label_seq(batch)

    pred_flow_seq = ensure_edge_seq(outputs.get("flow_pred", None))
    if pred_flow_seq is None:
        pred_flow_seq = ensure_edge_seq(outputs.get("flow_cond", None))

    edge_logits_seq = ensure_edge_seq(outputs.get("od_edge_logits", None))
    node_logits_seq = ensure_node_seq(outputs.get("od_node_logits", None))

    device = None
    for x in (pred_flow_seq, edge_logits_seq, node_logits_seq, y_flow_seq):
        if x is not None and torch.is_tensor(x):
            device = x.device
            break
    if device is None:
        device = torch.device("cpu")

    schedule = scheduled_base_weight(epoch=epoch, args=args)
    lambda_base_t = float(schedule["lambda_base_current"])
    lambda_comp = float(getattr(args, "lambda_comp", getattr(args, "peak_alpha", 1.0)))
    comp_weight_mode = str(getattr(args, "comp_weight_mode", getattr(args, "peak_weight_mode", "legacy"))).lower().strip()

    flow_loss_space = str(getattr(args, "flow_loss_space", "log1p")).lower().strip()
    flow_loss_type = str(getattr(args, "flow_loss_type", "smoothl1")).lower().strip()

    if pred_flow_seq is not None and y_flow_seq is not None and y_flow_seq.numel() > 0:
        L_base_raw = compensated_positive_flow_loss(
            pred_flow_seq,
            y_flow_seq,
            flow_loss_space=flow_loss_space,
            flow_loss_type=flow_loss_type,
            comp_weight_mode=comp_weight_mode,
            lambda_comp=lambda_comp,
        )
    else:
        L_base_raw = torch.tensor(0.0, device=device)

    L_base = lambda_base_t * L_base_raw

    if node_logits_seq is not None and y_node_seq is not None and y_node_seq.numel() > 0:
        node_losses = []
        h_max = min(node_logits_seq.shape[0], y_node_seq.shape[0])
        for h in range(h_max):
            node_losses.append(dynamic_bce_loss(node_logits_seq[h], y_node_seq[h]))
        L_node = torch.stack(node_losses).mean() if node_losses else torch.tensor(0.0, device=device)
    else:
        L_node = torch.tensor(0.0, device=device)

    if edge_logits_seq is not None and y_exist_seq is not None and y_exist_seq.numel() > 0:
        edge_losses = []
        num_nodes = int(batch["node_feat"].shape[0])
        edge_index = batch["edge_index"]
        last_flow = batch.get("last_flow", None)
        edge_neg_ratio = float(getattr(args, "edge_neg_ratio", 1.0))
        neg_active_only = bool(int(getattr(args, "neg_active_only", 1)))
        edge_loss_type = str(getattr(args, "edge_loss_type", getattr(args, "exist_loss_type", "focal"))).lower().strip()
        edge_alpha = float(getattr(args, "edge_alpha", getattr(args, "exist_alpha", 0.75)))
        edge_gamma = float(getattr(args, "edge_gamma", getattr(args, "exist_gamma", 2.0)))
        transition_boost = float(getattr(args, "transition_boost", 2.0))
        use_transition_weight = bool(int(getattr(args, "use_transition_weight", 1)))

        if last_flow is not None:
            prev_exist0 = (last_flow.view(-1) > 0).float()
        else:
            prev_exist0 = None

        h_max = min(edge_logits_seq.shape[0], y_exist_seq.shape[0])
        for h in range(h_max):
            sel = sample_edge_indices_for_bce(
                edge_index=edge_index,
                y_exist=y_exist_seq[h],
                last_flow=last_flow,
                num_nodes=num_nodes,
                neg_ratio=edge_neg_ratio,
                neg_active_only=neg_active_only,
            )
            if sel.numel() <= 0:
                continue

            if h == 0:
                y_prev_h = prev_exist0
            else:
                y_prev_h = y_exist_seq[h - 1].view(-1)

            logits_sel = edge_logits_seq[h].view(-1, 1)[sel]
            labels_sel = y_exist_seq[h].view(-1, 1)[sel]

            if use_transition_weight:
                weight_sel = build_transition_weight(
                    y_cur=y_exist_seq[h].view(-1, 1)[sel],
                    y_prev=None if y_prev_h is None else y_prev_h.view(-1, 1)[sel],
                    transition_boost=transition_boost,
                )
            else:
                weight_sel = None

            edge_losses.append(
                edge_activation_loss_with_logits(
                    logits_sel,
                    labels_sel,
                    loss_type=edge_loss_type,
                    alpha=edge_alpha,
                    gamma=edge_gamma,
                    sample_weight=weight_sel,
                )
            )

        L_edge = torch.stack(edge_losses).mean() if edge_losses else torch.tensor(0.0, device=device)
    else:
        L_edge = torch.tensor(0.0, device=device)

    if pred_flow_seq is not None and y_flow_seq is not None and y_flow_seq.numel() > 0:
        L_zero = hard_negative_zero_loss(
            pred_flow_seq,
            y_flow_seq,
            top_ratio=float(getattr(args, "zero_top_ratio", 0.01)),
            min_topk=int(getattr(args, "zero_min_topk", 64)),
            max_topk=int(getattr(args, "zero_max_topk", 2048)),
        )
        L_vol = volatility_smooth_l1_loss(
            pred_flow_seq,
            y_flow_seq,
            last_flow=batch.get("last_flow", None),
        )
        num_nodes = int(batch["node_feat"].shape[0])
        edge_src = batch.get("edge_src", batch["edge_index"][0])
        edge_dst = batch.get("edge_dst", batch["edge_index"][1])
        L_cons = od_conservation_loss(
            pred_flow_seq,
            y_flow_seq,
            edge_src=edge_src,
            edge_dst=edge_dst,
            num_nodes=num_nodes,
            out_weight=float(getattr(args, "cons_out_weight", 1.0)),
            in_weight=float(getattr(args, "cons_in_weight", 1.0)),
        )
    else:
        L_zero = torch.tensor(0.0, device=device)
        L_vol = torch.tensor(0.0, device=device)
        L_cons = torch.tensor(0.0, device=device)

    aux_scale = scheduled_scale(epoch, int(getattr(args, "aux_warmup_epochs", 15)))
    cons_scale = scheduled_scale(epoch, int(getattr(args, "cons_warmup_epochs", 10)))

    w_base = float(getattr(args, "w_base", getattr(args, "w_flow", 1.0)))
    w_zero = float(getattr(args, "w_zero", 0.035))
    w_vol = float(getattr(args, "w_vol", 0.1))
    w_cons = float(getattr(args, "w_cons", 0.1))
    w_node = float(getattr(args, "w_node", getattr(args, "w_od", 0.08)))
    w_edge = float(getattr(args, "w_edge", 0.08))

    L_struct = w_node * L_node + w_edge * L_edge
    total = (
        w_base * L_base
        + w_zero * L_zero
        + w_vol * L_vol
        + cons_scale * w_cons * L_cons
        + aux_scale * L_struct
    )

    return {
        "loss": total,
        "base": L_base.detach(),
        "base_raw": L_base_raw.detach(),
        "struct": L_struct.detach(),
        "node": L_node.detach(),
        "edge": L_edge.detach(),
        "zero": L_zero.detach(),
        "vol": L_vol.detach(),
        "cons": L_cons.detach(),
        "lambda_base": torch.tensor(lambda_base_t, device=device),
        "lambda_base_nominal": torch.tensor(float(schedule["lambda_base"]), device=device),
        "lambda_comp": torch.tensor(lambda_comp, device=device),
        "aux_scale": torch.tensor(float(aux_scale), device=device),
        "cons_scale": torch.tensor(float(cons_scale), device=device),

        # Compatibility keys for old logging code.
        "reg": L_base.detach(),
        "cond_pos": L_base_raw.detach(),
        "final_pos": L_base_raw.detach(),
        "lambda_cond": torch.tensor(float(schedule["lambda_base_cond_part"]), device=device),
        "lambda_final": torch.tensor(float(schedule["lambda_base_final_part"]), device=device),
    }
