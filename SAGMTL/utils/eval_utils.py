# utils/eval_utils.py
# -*- coding: utf-8 -*-
"""
Evaluation utilities for multi-step sparse OD forecasting.

This module provides:
- masked and all-edge regression metrics;
- node-level origin/destination activity metrics;
- edge existence metrics with threshold calibration;
- per-horizon metrics;
- zero-edge spill diagnostics;
- OD marginal consistency diagnostics.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _mask_np(array: np.ndarray, null_val, normalize: bool = True):
    if isinstance(null_val, float) and np.isnan(null_val):
        mask = ~np.isnan(array)
    else:
        mask = np.not_equal(array, null_val)

    mask = mask.astype(np.float32)
    if normalize:
        m = mask.mean()
        if m > 0:
            mask = mask / m
    return mask


def _binary_mask_np(array: np.ndarray, null_val):
    if isinstance(null_val, float) and np.isnan(null_val):
        return (~np.isnan(array)).astype(np.float32)
    return np.not_equal(array, null_val).astype(np.float32)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _as_pyobj(x):
    if isinstance(x, dict):
        return {k: _as_pyobj(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_as_pyobj(v) for v in x]
    if isinstance(x, tuple):
        return [_as_pyobj(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def masked_mae_np(y_true, y_pred, null_val=0.0):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    mask = _mask_np(y_true, null_val)
    err = np.abs(y_true - y_pred)
    return float(np.mean(np.nan_to_num(mask * err)))


def masked_mse_np(y_true, y_pred, null_val=0.0):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    mask = _mask_np(y_true, null_val)
    err = (y_true - y_pred) ** 2
    return float(np.mean(np.nan_to_num(mask * err)))


def masked_mape_np(
    y_true,
    y_pred,
    null_val=0.0,
    eps: float = 1e-8,
    mode: str = "clip",
):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    mask = _mask_np(y_true, null_val)

    y_abs = np.abs(y_true)
    if mode == "clip":
        denom = np.maximum(y_abs, eps)
    elif mode == "add":
        denom = y_abs + eps
    elif mode == "ignore_zeros":
        denom = np.where(y_abs < eps, np.nan, y_abs)
    else:
        raise ValueError(f"Unsupported mape mode: {mode}")

    mape = np.abs(y_pred - y_true) / denom
    mape = np.nan_to_num(mape)
    return float(np.mean(mask * mape) * 100.0)


def masked_smape_np(y_true, y_pred, null_val=0.0, eps: float = 1e-8):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    mask = _binary_mask_np(y_true, null_val)
    numerator = np.abs(y_pred - y_true) * mask
    denominator = (np.abs(y_true) + np.abs(y_pred) + eps) * mask
    smape = 200.0 * (numerator / denominator)
    valid = mask > 0
    if valid.any():
        return float(np.mean(smape[valid]))
    return 0.0


def masked_wape_np(y_true, y_pred, null_val=0.0, eps: float = 1e-8):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    mask = _binary_mask_np(y_true, null_val)
    numerator = np.sum(np.abs(y_pred - y_true) * mask)
    denominator = np.sum(np.abs(y_true) * mask) + eps
    return float(numerator / denominator * 100.0)


def reg_metrics(
    y_pred,
    y_true,
    masked: bool = True,
    null_val: float = 0.0,
    eps: float = 1e-8,
    mape_mode: str = "clip",
):
    y_true_np = _to_numpy(y_true).reshape(-1)
    y_pred_np = _to_numpy(y_pred).reshape(-1)

    if masked:
        mae_val = masked_mae_np(y_true_np, y_pred_np, null_val=null_val)
        mse_val = masked_mse_np(y_true_np, y_pred_np, null_val=null_val)
        rmse_val = float(np.sqrt(mse_val))
        mape_val = masked_mape_np(
            y_true_np,
            y_pred_np,
            null_val=null_val,
            eps=eps,
            mode=mape_mode,
        )
        return mae_val, rmse_val, mape_val

    diff = np.abs(y_pred_np - y_true_np)
    mae_val = float(np.mean(diff))
    rmse_val = float(np.sqrt(np.mean((y_pred_np - y_true_np) ** 2)))

    y_abs = np.abs(y_true_np)
    if mape_mode == "clip":
        denom = np.maximum(y_abs, eps)
    elif mape_mode == "add":
        denom = y_abs + eps
    elif mape_mode == "ignore_zeros":
        denom = np.where(y_abs < eps, np.nan, y_abs)
    else:
        raise ValueError(f"Unsupported mape mode: {mape_mode}")

    mape_val = float(np.nan_to_num(np.abs((y_pred_np - y_true_np) / denom)).mean() * 100.0)
    return mae_val, rmse_val, mape_val


def compute_cls_metrics(logits, y_true, threshold: float = 0.5):
    logits_np = _to_numpy(logits).reshape(-1)
    labels_np = _to_numpy(y_true).reshape(-1).astype(np.int32)
    prob = _sigmoid_np(logits_np)
    pred = (prob >= float(threshold)).astype(np.int32)

    acc = float((pred == labels_np).mean())
    tp = float(((pred == 1) & (labels_np == 1)).sum())
    fp = float(((pred == 1) & (labels_np == 0)).sum())
    fn = float(((pred == 0) & (labels_np == 1)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

    auc = None
    try:
        from sklearn.metrics import roc_auc_score

        if np.unique(labels_np).size >= 2:
            auc = float(roc_auc_score(labels_np, prob))
    except Exception:
        auc = None

    return {"acc": acc, "f1": float(f1), "auc": auc}


def compute_edge_exist_metrics(edge_logits, y_edge_exist, threshold: float = 0.5):
    m = compute_cls_metrics(edge_logits, y_edge_exist, threshold=threshold)
    return {"edge_acc": m["acc"], "edge_f1": m["f1"], "edge_auc": m["auc"]}


def mae(y_true, y_pred, null_val: float = 0.0):
    return masked_mae_np(y_true, y_pred, null_val=null_val)


def rmse(y_true, y_pred, null_val: float = 0.0):
    return float(np.sqrt(masked_mse_np(y_true, y_pred, null_val=null_val)))


def mape(y_true, y_pred, null_val: float = 0.0, eps: float = 1.0, mode: str = "clip"):
    return masked_mape_np(y_true, y_pred, null_val=null_val, eps=eps, mode=mode)


def compute_reg_metrics(y_pred, y_true):
    mae_val, rmse_val, _ = reg_metrics(
        y_pred,
        y_true,
        masked=True,
        null_val=0.0,
        eps=1.0,
        mape_mode="clip",
    )
    return {"mae": mae_val, "rmse": rmse_val}


def compute_metrics(y_pred, y_true, task: str = "reg"):
    if task == "reg":
        return compute_reg_metrics(y_pred, y_true)
    return compute_cls_metrics(y_pred, y_true)


def metrics_to_str(m: dict) -> str:
    return " ".join(
        [
            f"{k}={v:.4f}" if isinstance(v, (int, float)) and v == v else f"{k}=NA"
            for k, v in m.items()
        ]
    )


def _move_batch_to_device_local(batch: dict, device):
    out = {}
    for k, v in batch.items():
        if torch is not None and torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def ensure_edge_seq(x: Optional["torch.Tensor"]) -> Optional["torch.Tensor"]:
    if x is None:
        return None
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    raise ValueError(f"edge tensor must be 1D or 2D, got shape={tuple(x.shape)}")


def ensure_node_seq(x: Optional["torch.Tensor"]) -> Optional["torch.Tensor"]:
    if x is None:
        return None
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"node tensor must be 2D or 3D, got shape={tuple(x.shape)}")


def get_edge_labels_seq(batch: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
    if "edge_labels_seq" in batch:
        return ensure_edge_seq(batch["edge_labels_seq"].float())
    return ensure_edge_seq(batch["edge_labels"].float())


def get_y_exist_seq(batch: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
    if "y_od_seq" in batch:
        return ensure_edge_seq(batch["y_od_seq"].float())
    if "y_od" in batch:
        return ensure_edge_seq(batch["y_od"].float())
    y_edge = get_edge_labels_seq(batch)
    return (y_edge > 0.0).float()


def get_node_labels_seq(batch: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
    if "node_od_label_seq" in batch:
        return ensure_node_seq(batch["node_od_label_seq"].float())
    return ensure_node_seq(batch["node_od_label"].float())


def _compute_node_metrics_from_arrays(logits_np: np.ndarray, labels_np: np.ndarray, thr: float = 0.5):
    if logits_np.size == 0 or labels_np.size == 0:
        return 0.0, 0.0

    prob = _sigmoid_np(logits_np)
    pred = (prob >= thr).astype(np.int32)
    labels = labels_np.astype(np.int32)

    acc = float((pred == labels).astype(np.float32).mean())

    f1_list = []
    for c in range(pred.shape[1]):
        p = pred[:, c]
        t = labels[:, c]
        tp = ((p == 1) & (t == 1)).sum()
        fp = ((p == 1) & (t == 0)).sum()
        fn = ((p == 0) & (t == 1)).sum()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
        f1_list.append(float(f1))

    return acc, float(np.mean(f1_list)) if f1_list else 0.0


def _edge_cls_metrics_from_logits_labels(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    if logits.size == 0 or labels.size == 0:
        return {"edge_acc": 0.0, "edge_f1": 0.0, "edge_auc": float("nan")}

    y_true = np.asarray(labels).reshape(-1).astype(np.int32)
    y_prob = _sigmoid_np(np.asarray(logits).reshape(-1))
    y_pred = (y_prob >= float(threshold)).astype(np.int32)

    acc = float((y_pred == y_true).mean())
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = float(2.0 * precision * recall / (precision + recall + 1e-12))

    auc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        if np.unique(y_true).size >= 2:
            auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass

    return {"edge_acc": acc, "edge_f1": f1, "edge_auc": auc}


def _dual_reg_metrics(y_pred, y_true, null_val: float, eps: float):
    mae_masked, rmse_masked, mape_masked = reg_metrics(
        y_pred,
        y_true,
        masked=True,
        null_val=null_val,
        eps=eps,
        mape_mode="clip",
    )
    mae_all, rmse_all, mape_all = reg_metrics(
        y_pred,
        y_true,
        masked=False,
        null_val=null_val,
        eps=eps,
        mape_mode="clip",
    )
    return {
        "mae_masked": mae_masked,
        "rmse_masked": rmse_masked,
        "mape_masked": mape_masked,
        "mae_all": mae_all,
        "rmse_all": rmse_all,
        "mape_all": mape_all,
    }


def _consistency_metrics_from_arrays(
    pred_flow: np.ndarray,
    true_flow: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    num_nodes: int,
    eps: float = 1e-8,
) -> Dict[str, float]:
    if pred_flow.size == 0 or true_flow.size == 0:
        return {
            "cons_out_mae": 0.0,
            "cons_in_mae": 0.0,
            "cons_out_wape": 0.0,
            "cons_in_wape": 0.0,
        }

    pred_flow = np.asarray(pred_flow, dtype=np.float64)
    true_flow = np.asarray(true_flow, dtype=np.float64)
    horizon, _ = pred_flow.shape

    edge_src = edge_src.astype(np.int64).reshape(1, -1).repeat(horizon, axis=0)
    edge_dst = edge_dst.astype(np.int64).reshape(1, -1).repeat(horizon, axis=0)

    pred_out = np.zeros((horizon, num_nodes), dtype=np.float64)
    true_out = np.zeros((horizon, num_nodes), dtype=np.float64)
    pred_in = np.zeros((horizon, num_nodes), dtype=np.float64)
    true_in = np.zeros((horizon, num_nodes), dtype=np.float64)

    for h in range(horizon):
        np.add.at(pred_out[h], edge_src[h], pred_flow[h])
        np.add.at(true_out[h], edge_src[h], true_flow[h])
        np.add.at(pred_in[h], edge_dst[h], pred_flow[h])
        np.add.at(true_in[h], edge_dst[h], true_flow[h])

    diff_out = np.abs(pred_out - true_out)
    diff_in = np.abs(pred_in - true_in)

    return {
        "cons_out_mae": float(diff_out.mean()),
        "cons_in_mae": float(diff_in.mean()),
        "cons_out_wape": float(diff_out.sum() / (np.abs(true_out).sum() + eps) * 100.0),
        "cons_in_wape": float(diff_in.sum() / (np.abs(true_in).sum() + eps) * 100.0),
    }


def _zero_edge_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spill_thresholds=(0.1, 0.5),
) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mask_zero = y_true <= 0.0
    if mask_zero.sum() == 0:
        out = {"zero_mean_pred": 0.0, "zero_p95_pred": 0.0}
        for th in spill_thresholds:
            out[f"zero_spill_rate@{th:g}"] = 0.0
        return out

    pred_zero = y_pred[mask_zero]
    out = {
        "zero_mean_pred": float(pred_zero.mean()),
        "zero_p95_pred": float(np.quantile(pred_zero, 0.95)) if pred_zero.size > 0 else 0.0,
    }
    for th in spill_thresholds:
        out[f"zero_spill_rate@{th:g}"] = float((pred_zero > float(th)).mean())
    return out


def evaluate_multistep(
    model,
    loader,
    device,
    flow_loss_fn,
    dynamic_bce_loss_fn,
    sample_edge_indices_for_bce_fn,
    cons_loss_fn=None,
    thr_node: float = 0.5,
    edge_threshold: float = 0.5,
    neg_ratio: float = 1.0,
    neg_active_only: bool = True,
    mape_eps: float = 2.0,
    reg_masked: bool = True,
    reg_null_val: float = 0.0,
    w_flow: float = 1.0,
    w_cons: float = 0.0,
    w_od: float = 0.0,
    w_edge: float = 0.0,
    full_edge_eval: bool = False,
    spill_thresholds=(0.1, 0.5),
):
    if torch is None:
        raise RuntimeError("evaluate_multistep requires torch")

    model.eval()
    horizon = int(getattr(model, "horizon", 1))

    loss_sum = 0.0
    n_batches = 0

    flow_pred_rows = []
    flow_true_rows = []
    flow_pred_by_h = [[] for _ in range(horizon)]
    flow_true_by_h = [[] for _ in range(horizon)]

    node_logits_all = []
    node_labels_all = []

    edge_logits_eval_rows = []
    edge_labels_eval_rows = []
    edge_logits_eval_by_h = [[] for _ in range(horizon)]
    edge_labels_eval_by_h = [[] for _ in range(horizon)]

    cons_pred_rows = []
    cons_true_rows = []
    cons_pred_by_h = [[] for _ in range(horizon)]
    cons_true_by_h = [[] for _ in range(horizon)]
    edge_src_global = None
    edge_dst_global = None
    num_nodes_global = None

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device_local(batch, device)
            out = model(batch)

            y_edge_seq = get_edge_labels_seq(batch)
            y_exist_seq = get_y_exist_seq(batch)
            y_node_seq = get_node_labels_seq(batch)

            pred_edge_seq = ensure_edge_seq(out.get("flow_pred", None))
            edge_logits_seq = ensure_edge_seq(out.get("od_edge_logits", None))
            node_logits_seq = ensure_node_seq(out.get("od_node_logits", None))

            if pred_edge_seq is None:
                continue

            H = min(pred_edge_seq.shape[0], y_edge_seq.shape[0], horizon)
            pred_edge_seq = pred_edge_seq[:H]
            y_edge_seq = y_edge_seq[:H]
            y_exist_seq = y_exist_seq[:H]

            l_flow = flow_loss_fn(pred_edge_seq, y_edge_seq)

            if node_logits_seq is not None:
                Hn = min(node_logits_seq.shape[0], y_node_seq.shape[0], horizon)
                l_node_list = [
                    dynamic_bce_loss_fn(node_logits_seq[h], y_node_seq[h])
                    for h in range(Hn)
                ]
                l_node = torch.stack(l_node_list).mean() if l_node_list else torch.tensor(0.0, device=device)
            else:
                l_node = torch.tensor(0.0, device=device)

            if edge_logits_seq is not None:
                He = min(edge_logits_seq.shape[0], y_exist_seq.shape[0], horizon)
                l_edge_list = []

                for h in range(He):
                    if full_edge_eval:
                        sel = torch.arange(y_exist_seq[h].numel(), device=device, dtype=torch.long)
                    else:
                        sel = sample_edge_indices_for_bce_fn(
                            edge_index=batch["edge_index"],
                            y_exist=y_exist_seq[h],
                            last_flow=batch.get("last_flow", None),
                            num_nodes=int(batch["node_feat"].shape[0]),
                            neg_ratio=neg_ratio,
                            neg_active_only=neg_active_only,
                        )

                    if sel.numel() <= 0:
                        continue

                    logits_sel = edge_logits_seq[h].view(-1, 1)[sel]
                    labels_sel = y_exist_seq[h].view(-1, 1)[sel]
                    l_edge_list.append(dynamic_bce_loss_fn(logits_sel, labels_sel))

                l_edge = torch.stack(l_edge_list).mean() if l_edge_list else torch.tensor(0.0, device=device)
            else:
                l_edge = torch.tensor(0.0, device=device)

            if cons_loss_fn is not None:
                edge_src = batch.get("edge_src", batch["edge_index"][0])
                edge_dst = batch.get("edge_dst", batch["edge_index"][1])
                num_nodes = int(batch["node_feat"].shape[0])
                l_cons = cons_loss_fn(pred_edge_seq, y_edge_seq, edge_src, edge_dst, num_nodes)
            else:
                l_cons = torch.tensor(0.0, device=device)

            loss_ref = (
                float(w_flow) * l_flow
                + float(w_cons) * l_cons
                + float(w_od) * l_node
                + float(w_edge) * l_edge
            )
            loss_sum += float(loss_ref.item())
            n_batches += 1

            pred_np = pred_edge_seq.detach().cpu().numpy().astype(np.float64)
            true_np = y_edge_seq.detach().cpu().numpy().astype(np.float64)

            flow_pred_rows.append(pred_np)
            flow_true_rows.append(true_np)

            for h in range(H):
                flow_pred_by_h[h].append(pred_np[h])
                flow_true_by_h[h].append(true_np[h])

            if node_logits_seq is not None:
                Hn = min(node_logits_seq.shape[0], y_node_seq.shape[0], horizon)
                node_logits_all.append(node_logits_seq[:Hn].detach().cpu().numpy().reshape(-1, 2))
                node_labels_all.append(y_node_seq[:Hn].detach().cpu().numpy().reshape(-1, 2))

            if edge_logits_seq is not None:
                He = min(edge_logits_seq.shape[0], y_exist_seq.shape[0], horizon)
                for h in range(He):
                    logits_h = edge_logits_seq[h].detach().cpu().view(-1).numpy().astype(np.float64)
                    labels_h = y_exist_seq[h].detach().cpu().view(-1).numpy().astype(np.float64)
                    edge_logits_eval_rows.append(logits_h)
                    edge_labels_eval_rows.append(labels_h)
                    edge_logits_eval_by_h[h].append(logits_h)
                    edge_labels_eval_by_h[h].append(labels_h)

            cons_pred_rows.append(pred_np)
            cons_true_rows.append(true_np)
            for h in range(H):
                cons_pred_by_h[h].append(pred_np[h])
                cons_true_by_h[h].append(true_np[h])

            if edge_src_global is None:
                edge_src_global = _to_numpy(batch.get("edge_src", batch["edge_index"][0])).astype(np.int64)
                edge_dst_global = _to_numpy(batch.get("edge_dst", batch["edge_index"][1])).astype(np.int64)
                num_nodes_global = int(batch["node_feat"].shape[0])

    out = {"loss": float(loss_sum / max(n_batches, 1))}

    if flow_pred_rows:
        flow_pred_all_np = np.concatenate(flow_pred_rows, axis=0)
        flow_true_all_np = np.concatenate(flow_true_rows, axis=0)
    else:
        flow_pred_all_np = np.zeros((0, 0), dtype=np.float64)
        flow_true_all_np = np.zeros((0, 0), dtype=np.float64)

    out.update(_dual_reg_metrics(flow_pred_all_np, flow_true_all_np, null_val=reg_null_val, eps=mape_eps))

    if reg_masked:
        out["mae"] = out["mae_masked"]
        out["rmse"] = out["rmse_masked"]
        out["mape"] = out["mape_masked"]
    else:
        out["mae"] = out["mae_all"]
        out["rmse"] = out["rmse_all"]
        out["mape"] = out["mape_all"]

    if node_logits_all:
        node_logits_np = np.concatenate(node_logits_all, axis=0)
        node_labels_np = np.concatenate(node_labels_all, axis=0)
        node_acc, node_f1 = _compute_node_metrics_from_arrays(node_logits_np, node_labels_np, thr=thr_node)
    else:
        node_acc, node_f1 = 0.0, 0.0
    out["node_acc"] = float(node_acc)
    out["node_f1"] = float(node_f1)

    if edge_logits_eval_rows:
        edge_logits_np = np.concatenate(edge_logits_eval_rows, axis=0)
        edge_labels_np = np.concatenate(edge_labels_eval_rows, axis=0)
        edge_metrics = _edge_cls_metrics_from_logits_labels(edge_logits_np, edge_labels_np, threshold=edge_threshold)
    else:
        edge_metrics = {"edge_acc": 0.0, "edge_f1": 0.0, "edge_auc": float("nan")}
    out.update(edge_metrics)
    out["edge_threshold"] = float(edge_threshold)

    if cons_pred_rows and edge_src_global is not None and edge_dst_global is not None and num_nodes_global is not None:
        cons_pred_np = np.concatenate(cons_pred_rows, axis=0)
        cons_true_np = np.concatenate(cons_true_rows, axis=0)
        out.update(
            _consistency_metrics_from_arrays(
                cons_pred_np,
                cons_true_np,
                edge_src_global,
                edge_dst_global,
                num_nodes_global,
            )
        )
    else:
        out.update(
            {
                "cons_out_mae": 0.0,
                "cons_in_mae": 0.0,
                "cons_out_wape": 0.0,
                "cons_in_wape": 0.0,
            }
        )

    out.update(_zero_edge_metrics(flow_true_all_np, flow_pred_all_np, spill_thresholds=spill_thresholds))

    horizon_mae_masked = []
    horizon_rmse_masked = []
    horizon_mape_masked = []
    horizon_mae_all = []
    horizon_rmse_all = []
    horizon_mape_all = []
    horizon_edge_acc = []
    horizon_edge_f1 = []
    horizon_edge_auc = []
    horizon_cons_out_mae = []
    horizon_cons_in_mae = []
    horizon_cons_out_wape = []
    horizon_cons_in_wape = []
    horizon_zero_mean_pred = []
    horizon_zero_p95_pred = []

    for h in range(horizon):
        if flow_pred_by_h[h]:
            yp_h = np.stack(flow_pred_by_h[h], axis=0)
            yt_h = np.stack(flow_true_by_h[h], axis=0)
        else:
            yp_h = np.zeros((0, 0), dtype=np.float64)
            yt_h = np.zeros((0, 0), dtype=np.float64)

        reg_h = _dual_reg_metrics(yp_h, yt_h, null_val=reg_null_val, eps=mape_eps)
        horizon_mae_masked.append(float(reg_h["mae_masked"]))
        horizon_rmse_masked.append(float(reg_h["rmse_masked"]))
        horizon_mape_masked.append(float(reg_h["mape_masked"]))
        horizon_mae_all.append(float(reg_h["mae_all"]))
        horizon_rmse_all.append(float(reg_h["rmse_all"]))
        horizon_mape_all.append(float(reg_h["mape_all"]))

        if edge_logits_eval_by_h[h]:
            log_h = np.concatenate(edge_logits_eval_by_h[h], axis=0)
            lab_h = np.concatenate(edge_labels_eval_by_h[h], axis=0)
            edge_h = _edge_cls_metrics_from_logits_labels(log_h, lab_h, threshold=edge_threshold)
        else:
            edge_h = {"edge_acc": 0.0, "edge_f1": 0.0, "edge_auc": float("nan")}
        horizon_edge_acc.append(float(edge_h["edge_acc"]))
        horizon_edge_f1.append(float(edge_h["edge_f1"]))
        horizon_edge_auc.append(float(edge_h["edge_auc"]))

        if cons_pred_by_h[h] and edge_src_global is not None and edge_dst_global is not None and num_nodes_global is not None:
            cp = np.stack(cons_pred_by_h[h], axis=0)
            ct = np.stack(cons_true_by_h[h], axis=0)
            cons_h = _consistency_metrics_from_arrays(cp, ct, edge_src_global, edge_dst_global, num_nodes_global)
        else:
            cons_h = {
                "cons_out_mae": 0.0,
                "cons_in_mae": 0.0,
                "cons_out_wape": 0.0,
                "cons_in_wape": 0.0,
            }
        horizon_cons_out_mae.append(float(cons_h["cons_out_mae"]))
        horizon_cons_in_mae.append(float(cons_h["cons_in_mae"]))
        horizon_cons_out_wape.append(float(cons_h["cons_out_wape"]))
        horizon_cons_in_wape.append(float(cons_h["cons_in_wape"]))

        zero_h = _zero_edge_metrics(yt_h, yp_h, spill_thresholds=spill_thresholds)
        horizon_zero_mean_pred.append(float(zero_h["zero_mean_pred"]))
        horizon_zero_p95_pred.append(float(zero_h["zero_p95_pred"]))

    out["horizon_mae_masked"] = horizon_mae_masked
    out["horizon_rmse_masked"] = horizon_rmse_masked
    out["horizon_mape_masked"] = horizon_mape_masked
    out["horizon_mae_all"] = horizon_mae_all
    out["horizon_rmse_all"] = horizon_rmse_all
    out["horizon_mape_all"] = horizon_mape_all
    out["horizon_edge_acc"] = horizon_edge_acc
    out["horizon_edge_f1"] = horizon_edge_f1
    out["horizon_edge_auc"] = horizon_edge_auc
    out["horizon_cons_out_mae"] = horizon_cons_out_mae
    out["horizon_cons_in_mae"] = horizon_cons_in_mae
    out["horizon_cons_out_wape"] = horizon_cons_out_wape
    out["horizon_cons_in_wape"] = horizon_cons_in_wape
    out["horizon_zero_mean_pred"] = horizon_zero_mean_pred
    out["horizon_zero_p95_pred"] = horizon_zero_p95_pred

    for th in spill_thresholds:
        key = f"zero_spill_rate@{th:g}"
        vals = []
        for h in range(horizon):
            if flow_true_by_h[h]:
                yh = np.stack(flow_true_by_h[h], axis=0)
                ph = np.stack(flow_pred_by_h[h], axis=0)
                vals.append(float(_zero_edge_metrics(yh, ph, spill_thresholds=spill_thresholds)[key]))
            else:
                vals.append(0.0)
        out[f"horizon_{key}"] = vals

    return out


def calibrate_edge_threshold(
    model,
    loader,
    device,
    num_bins: int = 1000,
    min_threshold: float = 0.001,
    max_threshold: float = 0.999,
):
    if torch is None:
        raise RuntimeError("calibrate_edge_threshold requires torch")

    model.eval()

    num_bins = max(10, int(num_bins))
    min_threshold = max(0.0, min(1.0, float(min_threshold)))
    max_threshold = max(0.0, min(1.0, float(max_threshold)))
    if min_threshold > max_threshold:
        min_threshold, max_threshold = max_threshold, min_threshold

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    thresholds = bin_edges[:-1]

    pos_hist = np.zeros(num_bins, dtype=np.float64)
    neg_hist = np.zeros(num_bins, dtype=np.float64)

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device_local(batch, device)
            out = model(batch)

            y_exist_seq = get_y_exist_seq(batch)
            edge_logits_seq = ensure_edge_seq(out.get("od_edge_logits", None))
            if edge_logits_seq is None:
                continue

            H = min(edge_logits_seq.shape[0], y_exist_seq.shape[0])
            for h in range(H):
                logits_h = edge_logits_seq[h].detach().cpu().view(-1).numpy().astype(np.float64)
                labels_h = y_exist_seq[h].detach().cpu().view(-1).numpy().astype(np.int32)

                prob_h = _sigmoid_np(logits_h)
                pos_prob = prob_h[labels_h > 0]
                neg_prob = prob_h[labels_h <= 0]

                if pos_prob.size > 0:
                    pos_hist += np.histogram(pos_prob, bins=bin_edges)[0]
                if neg_prob.size > 0:
                    neg_hist += np.histogram(neg_prob, bins=bin_edges)[0]

    pos_total = float(pos_hist.sum())
    neg_total = float(neg_hist.sum())

    if pos_total <= 0.0 or neg_total <= 0.0:
        return {
            "edge_threshold": 0.5,
            "edge_f1": 0.0,
            "edge_precision": 0.0,
            "edge_recall": 0.0,
            "edge_f1_at_0.5": 0.0,
            "edge_precision_at_0.5": 0.0,
            "edge_recall_at_0.5": 0.0,
            "pos_total": pos_total,
            "neg_total": neg_total,
        }

    tp = np.cumsum(pos_hist[::-1])[::-1]
    fp = np.cumsum(neg_hist[::-1])[::-1]
    fn = pos_total - tp

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

    valid = (thresholds >= min_threshold) & (thresholds <= max_threshold)
    valid_idx = np.where(valid)[0]
    if valid_idx.size <= 0:
        best_i = int(np.nanargmax(f1))
    else:
        best_local = int(np.nanargmax(f1[valid_idx]))
        best_i = int(valid_idx[best_local])

    idx_05 = int(np.searchsorted(thresholds, 0.5, side="left"))
    idx_05 = min(max(idx_05, 0), len(thresholds) - 1)

    return {
        "edge_threshold": float(thresholds[best_i]),
        "edge_f1": float(f1[best_i]),
        "edge_precision": float(precision[best_i]),
        "edge_recall": float(recall[best_i]),
        "edge_f1_at_0.5": float(f1[idx_05]),
        "edge_precision_at_0.5": float(precision[idx_05]),
        "edge_recall_at_0.5": float(recall[idx_05]),
        "pos_total": pos_total,
        "neg_total": neg_total,
    }


def save_metrics_artifacts(run_dir: str, split: str, metrics: Dict[str, Any]):
    os.makedirs(run_dir, exist_ok=True)

    json_path = os.path.join(run_dir, f"{split}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_as_pyobj(metrics), f, ensure_ascii=False, indent=2)

    horizon_keys = [
        k for k, v in metrics.items()
        if isinstance(v, list) and k.startswith("horizon_")
    ]
    if not horizon_keys:
        return

    lengths = [len(metrics[k]) for k in horizon_keys if isinstance(metrics[k], list)]
    horizon = max(lengths) if lengths else 0
    if horizon <= 0:
        return

    csv_path = os.path.join(run_dir, f"{split}_horizon_metrics.csv")
    fieldnames = ["horizon"] + sorted(horizon_keys)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in range(horizon):
            row = {"horizon": h + 1}
            for k in horizon_keys:
                v = metrics.get(k, [])
                row[k] = v[h] if h < len(v) else ""
            writer.writerow(_as_pyobj(row))
