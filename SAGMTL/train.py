# train.py
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import math
import random
import warnings
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from config import get_args

if os.name == "nt":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    if "--num_workers" in sys.argv and any(v not in ("0", "1") for v in sys.argv):
        warnings.warn(
            "[data] Multiprocessing on Windows may be unstable; use --num_workers 0 or 1.",
            stacklevel=1,
        )

from model import ODModel
from utils.data_utils import get_hour_od_dataloaders, move_batch_to_device
from utils.eval_utils import (
    reg_metrics,
    ensure_edge_seq,
    get_edge_labels_seq,
    evaluate_multistep,
    calibrate_edge_threshold,
    save_metrics_artifacts,
)
from utils.loss_utils import (
    compute_train_loss,
    log_loss_config,
    compensated_positive_flow_loss,
    dynamic_bce_loss,
    sample_edge_indices_for_bce,
    od_conservation_loss,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(params, lr: float, wd: float):
    return AdamW(params, lr=lr, weight_decay=wd)


class EarlyStopper:
    def __init__(self, patience: int = 10):
        self.best = math.inf
        self.count = 0
        self.patience = int(patience)

    def step(self, cur: float) -> bool:
        improved = cur < self.best
        if improved:
            self.best = cur
            self.count = 0
        else:
            self.count += 1
        return improved

    def should_stop(self) -> bool:
        return self.count >= self.patience


def setup_logging(log_dir: str, filename: str = "train.log"):
    import logging

    os.makedirs(log_dir, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(log_dir, filename), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.addHandler(fh)
    root.addHandler(sh)
    return root


def _get_device(args):
    use_cuda = (str(args.device).lower() == "cuda") and torch.cuda.is_available()
    if not use_cuda:
        return torch.device("cpu"), False, "cpu"
    local_gpu = int(getattr(args, "local_gpu", 0))
    return torch.device(f"cuda:{local_gpu}"), True, f"cuda:{local_gpu}"


def _compute_composite_early_stop_score(metrics: Dict[str, float]) -> float:
    mae_masked = float(metrics.get("mae_masked", math.inf))
    mae_all = float(metrics.get("mae_all", math.inf))
    edge_f1 = float(metrics.get("edge_f1", 0.0))
    cons_out_wape = float(metrics.get("cons_out_wape", math.inf))
    cons_in_wape = float(metrics.get("cons_in_wape", math.inf))

    if not math.isfinite(mae_masked) or not math.isfinite(mae_all):
        return math.inf
    if not math.isfinite(cons_out_wape) or not math.isfinite(cons_in_wape):
        return math.inf

    edge_f1 = max(0.0, min(1.0, edge_f1))
    cons_mean_scaled = (cons_out_wape + cons_in_wape) / 200.0
    return float(
        1.0 * mae_masked
        + 0.5 * (10.0 * mae_all)
        + 0.7 * (1.0 - edge_f1)
        + 0.3 * cons_mean_scaled
    )


def _resolve_early_stop_value(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name == "mae_masked":
        return float(metrics["mae_masked"])
    if metric_name == "mae_all":
        return float(metrics["mae_all"])
    if metric_name == "horizon_mae_masked_mean":
        vals = metrics.get("horizon_mae_masked", [])
        return float(np.mean(vals)) if vals else math.inf
    if metric_name == "horizon_mae_all_mean":
        vals = metrics.get("horizon_mae_all", [])
        return float(np.mean(vals)) if vals else math.inf
    if metric_name == "composite_v1":
        return _compute_composite_early_stop_score(metrics)
    raise ValueError(f"Unsupported early_stop_metric: {metric_name}")


def _item(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if torch.is_tensor(x):
        return float(x.detach().float().mean().item())
    try:
        return float(x)
    except Exception:
        return float(default)


def _make_flow_loss_fn(args):
    lambda_comp = float(getattr(args, "lambda_comp", getattr(args, "peak_alpha", 1.0)))
    comp_weight_mode = str(
        getattr(args, "comp_weight_mode", getattr(args, "peak_weight_mode", "legacy"))
    ).lower().strip()
    flow_loss_space = str(getattr(args, "flow_loss_space", "log1p")).lower().strip()
    flow_loss_type = str(getattr(args, "flow_loss_type", "smoothl1")).lower().strip()

    def flow_loss_fn(pred, target):
        return compensated_positive_flow_loss(
            pred,
            target,
            flow_loss_space=flow_loss_space,
            flow_loss_type=flow_loss_type,
            comp_weight_mode=comp_weight_mode,
            lambda_comp=lambda_comp,
        )

    return flow_loss_fn


def _make_cons_loss_fn(args):
    def cons_loss_fn(pred_flow, true_flow, edge_src, edge_dst, num_nodes):
        return od_conservation_loss(
            pred_flow,
            true_flow,
            edge_src=edge_src,
            edge_dst=edge_dst,
            num_nodes=num_nodes,
            out_weight=float(getattr(args, "cons_out_weight", 1.0)),
            in_weight=float(getattr(args, "cons_in_weight", 1.0)),
        )

    return cons_loss_fn


def _log_runtime_switches(logger, args, horizon: int) -> None:
    logger.info(
        "[model-switches] "
        f"use_edge_history={getattr(args, 'use_edge_history', 1)} "
        f"edge_history_input={getattr(args, 'edge_history_input', 'flow_exist')} "
        f"edge_history_dim={getattr(args, 'edge_history_dim', getattr(args, 'hidden_dim', 96))} "
        f"edge_history_fuse={getattr(args, 'edge_history_fuse', 'concat')} | "
        f"num_co_layers={getattr(args, 'num_co_layers', 2)} "
        f"co_channels={getattr(args, 'co_channels', 4)} "
        f"use_directional_co={getattr(args, 'use_directional_co', 0)} | "
        f"use_time_dyn={getattr(args, 'use_time_dyn', 1)} "
        f"use_static_node={getattr(args, 'use_static_node', 1)} "
        f"static_dyn_fuse_mode={getattr(args, 'static_dyn_fuse_mode', 'concat')} | "
        f"use_spatial_attn={getattr(args, 'use_spatial_attn', 1)} "
        f"spatial_dist_path={getattr(args, 'spatial_dist_path', '')} "
        f"spatial_attn_heads={getattr(args, 'spatial_attn_heads', 4)} "
        f"spatial_attn_res_weight={float(getattr(args, 'spatial_attn_res_weight', 0.15)):.3f} | "
        f"dyn_decoder_mode={getattr(args, 'dyn_decoder_mode', 'residual')} "
        f"dyn_res_use_tanh={getattr(args, 'dyn_res_use_tanh', 1)} "
        f"dyn_res_scale={float(getattr(args, 'dyn_res_scale', 1.0)):.3f} | "
        f"use_edge_history_in_decoder={getattr(args, 'use_edge_history_in_decoder', 1)} "
        f"static_branch_mode={getattr(args, 'static_branch_mode', 'static_roles')} "
        f"sd_fusion_mode={getattr(args, 'sd_fusion_mode', 'adaptive')} | "
        f"use_static_basis_head={getattr(args, 'use_static_basis_head', 0)} "
        f"use_dyn_query_head={getattr(args, 'use_dyn_query_head', 0)} | "
        f"dyn_share_floor={float(getattr(args, 'dyn_share_floor', 0.35)):.3f} "
        f"static_scale={float(getattr(args, 'static_scale', 0.60)):.3f} "
        f"horizon={horizon}"
    )


def _spatial_attention_sanity_check(logger, args) -> None:
    use_spatial_attn = bool(int(getattr(args, "use_spatial_attn", 0)))
    spatial_dist_path = str(getattr(args, "spatial_dist_path", "")).strip()

    if not use_spatial_attn:
        logger.info("[spatial-attn] disabled")
        return

    if not spatial_dist_path:
        raise ValueError(
            "use_spatial_attn=1 requires a valid --spatial_dist_path pointing to spatial_dist.npy."
        )
    if not os.path.isfile(spatial_dist_path):
        raise FileNotFoundError(f"spatial_dist_path does not exist: {spatial_dist_path}")

    logger.info(
        "[spatial-attn] enabled | "
        f"spatial_dist_path={spatial_dist_path} | "
        f"heads={getattr(args, 'spatial_attn_heads', 4)} | "
        f"dropout={float(getattr(args, 'spatial_attn_dropout', getattr(args, 'enc_dropout', 0.1))):.3f} | "
        f"res_weight={float(getattr(args, 'spatial_attn_res_weight', 0.15)):.3f} | "
        f"exclude_self={getattr(args, 'spatial_attn_exclude_self', 1)} | "
        f"dist_decay={getattr(args, 'use_spatial_dist_decay', 1)} | "
        f"dist_decay_init={float(getattr(args, 'spatial_dist_decay_init', 0.1)):.3f} | "
        f"dist_norm={getattr(args, 'spatial_dist_norm', 'max')}"
    )


def _log_train_epoch(
    logger,
    epoch: int,
    epochs: int,
    elapsed: float,
    lr: float,
    train_stats: Dict[str, float],
    freeze_static_now: bool,
) -> None:
    logger.info(
        f"[Train] Epoch {epoch}/{epochs} time={elapsed:.2f}s lr={lr:.3e}  "
        f"train_loss: {train_stats['loss']:.4f}  "
        f"base: {train_stats['base']:.4f}  base_raw: {train_stats['base_raw']:.4f}  "
        f"zero: {train_stats['zero']:.4f}  vol: {train_stats['vol']:.4f}  "
        f"cons: {train_stats['cons']:.4f}  struct: {train_stats['struct']:.4f}  "
        f"node: {train_stats['node']:.4f}  edge: {train_stats['edge']:.4f}  "
        f"lambda_base={train_stats['lambda_base']:.3f}  "
        f"lambda_comp={train_stats['lambda_comp']:.3f}  "
        f"aux_scale={train_stats['aux_scale']:.3f}  cons_scale={train_stats['cons_scale']:.3f}  "
        f"freeze_static={int(freeze_static_now)}"
    )


def main():
    import logging

    args = get_args()
    set_seed(int(args.seed))

    device, use_cuda, device_desc = _get_device(args)

    run_dir = args.exp_dir
    logger = setup_logging(run_dir, "train.log")
    logger.info(f"Outputs will be saved under: {run_dir}")
    logger.info(f"Using device: {device_desc}")

    try:
        with open(os.path.join(run_dir, "args_runtime.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save args_runtime.json: {e}")

    horizon = int(getattr(args, "horizon", 1))
    step_ahead = int(getattr(args, "step_ahead", 1))

    loaders, meta = get_hour_od_dataloaders(
        od_root=args.od_root,
        T_in=args.T_in,
        time_dim=args.time_dim,
        step_ahead=step_ahead,
        horizon=horizon,
        batch_size=int(args.batch_size),
        num_workers=0 if os.name == "nt" else int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        stride_train=args.stride_train,
        stride_eval=args.stride_eval,
        use_last_flow_gate=bool(args.use_last_flow_gate),
        edge_neg_ratio=float(getattr(args, "edge_neg_ratio", 1.0)),
        neg_active_only=bool(int(getattr(args, "neg_active_only", 1))),
    )

    logger.info(
        f"[data] dataset={args.dataset_name} N={meta['N']} F={meta['F']} T_total={meta['T_total']} "
        f"train#={meta['train_size']} val#={meta['val_size']} test#={meta['test_size']}"
    )
    logger.info(f"[data] static E_all={meta.get('E_all', -1)} global_candidate_edges")
    logger.info(
        f"[horizon] step_ahead={meta.get('step_ahead')} horizon={meta.get('horizon')} "
        f"predict_horizon={meta.get('predict_horizon')}"
    )

    edge_neg_ratio = float(getattr(args, "edge_neg_ratio", 1.0))
    neg_active_only = bool(int(getattr(args, "neg_active_only", 1)))
    logger.info(f"[neg-sampling] edge_neg_ratio={edge_neg_ratio} neg_active_only={neg_active_only}")

    reg_masked = bool(int(getattr(args, "reg_masked", 1)))
    reg_null_val = float(getattr(args, "reg_null_val", 0.0))
    mape_eps = float(getattr(args, "mape_eps", 2.0))
    logger.info(
        f"[metric] reg_masked={reg_masked} reg_null_val={reg_null_val} "
        f"mape_eps={mape_eps} early_stop_metric={args.early_stop_metric}"
    )

    _spatial_attention_sanity_check(logger, args)
    _log_runtime_switches(logger, args, horizon=horizon)

    model = ODModel(args).to(device)

    try:
        logger.info(f"[model] {model}")
    except Exception:
        pass

    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Params: total={total_params:,} trainable={trainable_params:,}")
    except Exception:
        pass

    log_loss_config(logger, args)

    flow_loss_fn = _make_flow_loss_fn(args)
    cons_loss_fn = _make_cons_loss_fn(args)

    optimizer = build_optimizer(model.parameters(), args.lr, args.weight_decay)

    stopper = EarlyStopper(patience=int(args.patience))
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")

    epochs = int(args.epochs)
    best_epoch = -1
    best_val_score = math.inf
    train_metric_max_batches = int(getattr(args, "train_metric_max_batches", 8))

    for epoch in range(1, epochs + 1):
        model.train()
        tic = time.time()

        freeze_static_epochs = int(getattr(args, "freeze_static_epochs", 0))
        freeze_static_now = epoch <= freeze_static_epochs
        for name, p in model.named_parameters():
            if ("flow_static_head" in name) or ("static_gate" in name) or ("static_bias" in name):
                p.requires_grad = not freeze_static_now

        sums = {
            "loss": torch.zeros((), device=device),
            "base": torch.zeros((), device=device),
            "base_raw": torch.zeros((), device=device),
            "zero": torch.zeros((), device=device),
            "vol": torch.zeros((), device=device),
            "cons": torch.zeros((), device=device),
            "struct": torch.zeros((), device=device),
            "node": torch.zeros((), device=device),
            "edge": torch.zeros((), device=device),
            "lambda_base": torch.zeros((), device=device),
            "lambda_comp": torch.zeros((), device=device),
            "aux_scale": torch.zeros((), device=device),
            "cons_scale": torch.zeros((), device=device),
        }
        n_step = 0

        tr_pred_flow: List[torch.Tensor] = []
        tr_true_flow: List[torch.Tensor] = []

        diag_alpha_list: List[torch.Tensor] = []
        diag_fs_list: List[torch.Tensor] = []
        diag_fd_list: List[torch.Tensor] = []

        for batch_idx, batch in enumerate(loaders["train"], start=1):
            batch = move_batch_to_device(batch, device)
            out = model(batch)

            loss_dict = compute_train_loss(
                outputs=out,
                batch=batch,
                args=args,
                epoch=epoch,
            )
            loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if getattr(args, "grad_clip", 0) and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            for key in sums.keys():
                if key in loss_dict:
                    sums[key] = sums[key] + loss_dict[key].detach()
            n_step += 1

            y_edge_seq = get_edge_labels_seq(batch)
            y_edge_pred_seq = ensure_edge_seq(out.get("flow_pred", None))
            use_this_batch_for_train_metric = (
                train_metric_max_batches <= 0
                or batch_idx <= train_metric_max_batches
            )
            if use_this_batch_for_train_metric and y_edge_pred_seq is not None and y_edge_seq.numel() > 0:
                tr_pred_flow.append(y_edge_pred_seq.detach().reshape(-1).float().cpu())
                tr_true_flow.append(y_edge_seq.detach().reshape(-1).float().cpu())

            alpha = ensure_edge_seq(out.get("alpha_sd", None))
            fs = ensure_edge_seq(out.get("flow_static", None))
            fd = ensure_edge_seq(out.get("flow_dyn", None))
            if (
                alpha is not None
                and fs is not None
                and fd is not None
                and alpha.numel() > 0
                and len(diag_alpha_list) < 5
            ):
                a_flat = alpha.detach().reshape(-1)
                fs_flat = fs.detach().reshape(-1)
                fd_flat = fd.detach().reshape(-1)
                n = int(a_flat.numel())
                k = min(n, 512)
                if k > 0:
                    idx = torch.randperm(n, device=a_flat.device)[:k]
                    diag_alpha_list.append(a_flat[idx].float().cpu())
                    diag_fs_list.append(fs_flat[idx].float().cpu())
                    diag_fd_list.append(fd_flat[idx].float().cpu())

        denom = max(1, n_step)
        train_stats = {key: float((val / denom).item()) for key, val in sums.items()}
        cur_lr = optimizer.param_groups[0].get("lr", args.lr)

        _log_train_epoch(
            logger=logger,
            epoch=epoch,
            epochs=epochs,
            elapsed=time.time() - tic,
            lr=cur_lr,
            train_stats=train_stats,
            freeze_static_now=freeze_static_now,
        )

        if tr_pred_flow:
            t_pred = torch.cat(tr_pred_flow, dim=0).numpy()
            t_true = torch.cat(tr_true_flow, dim=0).numpy()
            t_mae_masked, t_rmse_masked, t_mape_masked = reg_metrics(
                t_pred,
                t_true,
                masked=True,
                null_val=reg_null_val,
                eps=mape_eps,
                mape_mode="clip",
            )
            t_mae_all, t_rmse_all, t_mape_all = reg_metrics(
                t_pred,
                t_true,
                masked=False,
                null_val=reg_null_val,
                eps=mape_eps,
                mape_mode="clip",
            )
            logger.info(
                f"[Train-Metrics-sampled] masked(mae/rmse/mape)=({t_mae_masked:.4f}, {t_rmse_masked:.4f}, {t_mape_masked:.4f})  "
                f"all(mae/rmse/mape)=({t_mae_all:.4f}, {t_rmse_all:.4f}, {t_mape_all:.4f})"
            )

        w_node_eval = float(getattr(args, "w_node", getattr(args, "w_od", 0.08)))
        w_edge_eval = float(getattr(args, "w_edge", 0.08))

        val_metrics = evaluate_multistep(
            model=model,
            loader=loaders["val"],
            device=device,
            flow_loss_fn=flow_loss_fn,
            dynamic_bce_loss_fn=dynamic_bce_loss,
            sample_edge_indices_for_bce_fn=sample_edge_indices_for_bce,
            cons_loss_fn=cons_loss_fn,
            thr_node=float(getattr(args, "eval_threshold", 0.5)),
            neg_ratio=edge_neg_ratio,
            neg_active_only=neg_active_only,
            mape_eps=mape_eps,
            reg_masked=reg_masked,
            reg_null_val=reg_null_val,
            w_flow=float(getattr(args, "w_base", getattr(args, "w_flow", 1.0))),
            w_cons=float(getattr(args, "w_cons", 0.1)),
            w_od=w_node_eval,
            w_edge=w_edge_eval,
            full_edge_eval=True,
        )

        cur_score = _resolve_early_stop_value(val_metrics, args.early_stop_metric)

        improved = cur_score < best_val_score
        if improved:
            best_val_score = cur_score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_score": best_val_score,
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                best_path,
            )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_score": best_val_score,
                "args": vars(args),
                "val_metrics": val_metrics,
            },
            last_path,
        )

        if args.early_stop_metric == "composite_v1":
            mae_masked = float(val_metrics.get("mae_masked", math.inf))
            mae_all = float(val_metrics.get("mae_all", math.inf))
            edge_f1 = float(val_metrics.get("edge_f1", 0.0))
            cons_out_wape = float(val_metrics.get("cons_out_wape", math.inf))
            cons_in_wape = float(val_metrics.get("cons_in_wape", math.inf))
            cons_mean_scaled = (cons_out_wape + cons_in_wape) / 200.0
            logger.info(
                f"[Val-EarlyStop-composite_v1] score={cur_score:.4f} | "
                f"mae_masked={mae_masked:.4f} | "
                f"10*mae_all={10.0 * mae_all:.4f} | "
                f"1-edge_f1={1.0 - edge_f1:.4f} | "
                f"cons_mean_scaled={cons_mean_scaled:.4f}"
            )

        if improved:
            best_tag = (
                " (best by composite_v1)"
                if args.early_stop_metric == "composite_v1"
                else f" (best by {args.early_stop_metric})"
            )
        else:
            best_tag = f" (best @ {best_epoch} by {args.early_stop_metric})"

        logger.info(
            f"[Val] loss(ref): {val_metrics['loss']:.4f}  "
            f"masked(mae/rmse/mape)=({val_metrics['mae_masked']:.4f}, {val_metrics['rmse_masked']:.4f}, {val_metrics['mape_masked']:.4f})  "
            f"all(mae/rmse/mape)=({val_metrics['mae_all']:.4f}, {val_metrics['rmse_all']:.4f}, {val_metrics['mape_all']:.4f})  "
            f"node_f1={val_metrics['node_f1']:.4f} edge_f1={val_metrics['edge_f1']:.4f}  "
            f"cons_out/in_mae=({val_metrics.get('cons_out_mae', 0.0):.4f}/{val_metrics.get('cons_in_mae', 0.0):.4f})"
            f"{best_tag}"
        )

        try:
            H_show = min(3, len(val_metrics.get("horizon_mae_masked", [])))
            if H_show > 0:
                horizon_rows = []
                for h in range(H_show):
                    horizon_rows.append(
                        f"h{h + 1}: mae_masked={val_metrics['horizon_mae_masked'][h]:.4f}, "
                        f"mae_all={val_metrics['horizon_mae_all'][h]:.4f}, "
                        f"edge_f1={val_metrics['horizon_edge_f1'][h]:.4f}, "
                        f"cons_out/in=({val_metrics.get('horizon_cons_out_mae', [0.0] * H_show)[h]:.4f}/"
                        f"{val_metrics.get('horizon_cons_in_mae', [0.0] * H_show)[h]:.4f})"
                    )
                logger.info("[Val-Horizon] " + " | ".join(horizon_rows))
        except Exception:
            pass

        if diag_alpha_list:
            a_all = torch.cat(diag_alpha_list, dim=0).numpy()
            fs_all = torch.cat(diag_fs_list, dim=0).numpy()
            fd_all = torch.cat(diag_fd_list, dim=0).numpy()
            corr = float(np.corrcoef(fs_all, fd_all)[0, 1]) if fs_all.size > 1 else float("nan")
            logger.info(
                "[Diag] alpha_sd: mean=%.4f std=%.4f frac>0.7=%.4f frac<0.3=%.4f | E|flow_static|=%.4f E|flow_dyn|=%.4f corr(fs,fd)=%.4f"
                % (
                    float(a_all.mean()),
                    float(a_all.std()),
                    float((a_all > 0.7).mean()),
                    float((a_all < 0.3).mean()),
                    float(np.abs(fs_all).mean()),
                    float(np.abs(fd_all).mean()),
                    corr,
                )
            )

        stopper.step(cur_score)
        if stopper.should_stop():
            logger.info("Early stopping triggered.")
            break

    if os.path.isfile(best_path):
        state = torch.load(best_path, map_location="cpu")
        model.load_state_dict(state["model"])
        logging.getLogger().info(f"Loading best checkpoint for test: {best_path}")
    else:
        state = torch.load(last_path, map_location="cpu")
        model.load_state_dict(state["model"])
        logging.getLogger().info(f"No best checkpoint found; using last checkpoint: {last_path}")

    edge_thr_info = calibrate_edge_threshold(
        model=model,
        loader=loaders["val"],
        device=device,
        num_bins=1000,
        min_threshold=0.001,
        max_threshold=0.999,
    )
    edge_eval_threshold = float(edge_thr_info["edge_threshold"])

    logging.getLogger().info(
        "[EdgeThreshold-Calib] "
        f"best_thr={edge_eval_threshold:.4f} | "
        f"val_edge_f1_best={edge_thr_info['edge_f1']:.4f} | "
        f"val_edge_precision_best={edge_thr_info['edge_precision']:.4f} | "
        f"val_edge_recall_best={edge_thr_info['edge_recall']:.4f} | "
        f"val_edge_f1@0.5={edge_thr_info['edge_f1_at_0.5']:.4f} | "
        f"val_edge_precision@0.5={edge_thr_info['edge_precision_at_0.5']:.4f} | "
        f"val_edge_recall@0.5={edge_thr_info['edge_recall_at_0.5']:.4f} | "
        f"pos={edge_thr_info['pos_total']:.0f} neg={edge_thr_info['neg_total']:.0f}"
    )

    w_node_eval = float(getattr(args, "w_node", getattr(args, "w_od", 0.08)))
    w_edge_eval = float(getattr(args, "w_edge", 0.08))

    test_metrics = evaluate_multistep(
        model=model,
        loader=loaders["test"],
        device=device,
        flow_loss_fn=flow_loss_fn,
        dynamic_bce_loss_fn=dynamic_bce_loss,
        sample_edge_indices_for_bce_fn=sample_edge_indices_for_bce,
        cons_loss_fn=cons_loss_fn,
        thr_node=float(getattr(args, "eval_threshold", 0.5)),
        edge_threshold=edge_eval_threshold,
        neg_ratio=edge_neg_ratio,
        neg_active_only=neg_active_only,
        mape_eps=mape_eps,
        reg_masked=reg_masked,
        reg_null_val=reg_null_val,
        w_flow=float(getattr(args, "w_base", getattr(args, "w_flow", 1.0))),
        w_cons=float(getattr(args, "w_cons", 0.1)),
        w_od=w_node_eval,
        w_edge=w_edge_eval,
        full_edge_eval=True,
    )

    logging.getLogger().info("\n" + "=" * 100)
    logging.getLogger().info(" " * 35 + "TEST OVERALL METRICS" + " " * 35)
    logging.getLogger().info("=" * 100)
    logging.getLogger().info(
        f"[Test] loss={test_metrics['loss']:.4f} | "
        f"Primary(mae/rmse/mape)=({test_metrics['mae']:.4f}/{test_metrics['rmse']:.4f}/{test_metrics['mape']:.4f}) | "
        f"Masked(mae/rmse/mape)=({test_metrics['mae_masked']:.4f}/{test_metrics['rmse_masked']:.4f}/{test_metrics['mape_masked']:.4f}) | "
        f"All(mae/rmse/mape)=({test_metrics['mae_all']:.4f}/{test_metrics['rmse_all']:.4f}/{test_metrics['mape_all']:.4f})"
    )
    edge_auc_test = test_metrics["edge_auc"]
    edge_auc_test_str = f"{edge_auc_test:.4f}" if edge_auc_test == edge_auc_test else "nan"
    logging.getLogger().info(
        f"Node ACC/F1=({test_metrics['node_acc']:.4f}/{test_metrics['node_f1']:.4f}) | "
        f"Edge ACC/F1/AUC(full-eval)=({test_metrics['edge_acc']:.4f}/{test_metrics['edge_f1']:.4f}/{edge_auc_test_str}) | "
        f"edge_thr={test_metrics.get('edge_threshold', 0.5):.4f}"
    )
    logging.getLogger().info(
        f"Consistency OUT(intra-node agg) MAE/WAPE=({test_metrics.get('cons_out_mae', 0.0):.4f}/{test_metrics.get('cons_out_wape', 0.0):.4f}) | "
        f"Consistency IN(intra-node agg) MAE/WAPE=({test_metrics.get('cons_in_mae', 0.0):.4f}/{test_metrics.get('cons_in_wape', 0.0):.4f})"
    )
    logging.getLogger().info(
        f"Zero-edge mean/p95 pred=({test_metrics.get('zero_mean_pred', 0.0):.4f}/{test_metrics.get('zero_p95_pred', 0.0):.4f}) | "
        f"Zero spill@0.1={test_metrics.get('zero_spill_rate@0.1', 0.0):.4f} "
        f"spill@0.5={test_metrics.get('zero_spill_rate@0.5', 0.0):.4f}"
    )

    try:
        H_total = len(test_metrics.get("horizon_mae_masked", []))
        if H_total > 0:
            logging.getLogger().info("\n" + "-" * 130)
            logging.getLogger().info(" " * 47 + "PER HORIZON METRICS" + " " * 47)
            logging.getLogger().info("-" * 130)
            for h in range(H_total):
                edge_auc_h = test_metrics["horizon_edge_auc"][h]
                edge_auc_h_str = f"{edge_auc_h:.4f}" if edge_auc_h == edge_auc_h else "nan"
                logging.getLogger().info(
                    f"[Horizon h{h + 1}] "
                    f"MAE(masked/all)=({test_metrics['horizon_mae_masked'][h]:.4f}/{test_metrics['horizon_mae_all'][h]:.4f}) | "
                    f"RMSE(masked/all)=({test_metrics['horizon_rmse_masked'][h]:.4f}/{test_metrics['horizon_rmse_all'][h]:.4f}) | "
                    f"MAPE(masked/all)=({test_metrics['horizon_mape_masked'][h]:.4f}/{test_metrics['horizon_mape_all'][h]:.4f}) | "
                    f"Edge ACC/F1/AUC=({test_metrics['horizon_edge_acc'][h]:.4f}/{test_metrics['horizon_edge_f1'][h]:.4f}/{edge_auc_h_str}) | "
                    f"Cons OUT MAE/WAPE=({test_metrics.get('horizon_cons_out_mae', [0.0] * H_total)[h]:.4f}/{test_metrics.get('horizon_cons_out_wape', [0.0] * H_total)[h]:.4f}) | "
                    f"Cons IN MAE/WAPE=({test_metrics.get('horizon_cons_in_mae', [0.0] * H_total)[h]:.4f}/{test_metrics.get('horizon_cons_in_wape', [0.0] * H_total)[h]:.4f}) | "
                    f"Zero mean/p95=({test_metrics.get('horizon_zero_mean_pred', [0.0] * H_total)[h]:.4f}/{test_metrics.get('horizon_zero_p95_pred', [0.0] * H_total)[h]:.4f})"
                )
            logging.getLogger().info("-" * 130 + "\n")
    except Exception as e:
        logging.getLogger().warning(f"Failed to print per-horizon metrics: {e}")

    save_metrics_artifacts(run_dir, "test", test_metrics)

    logging.getLogger().info("Training complete.")
    logging.getLogger().info(f"Best val {args.early_stop_metric}: {best_val_score:.4f} best_epoch: {best_epoch}")
    logging.getLogger().info(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
