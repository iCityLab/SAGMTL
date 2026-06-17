# config.py
# -*- coding: utf-8 -*-
import os
import argparse
import json
import re
from datetime import datetime


def _default_num_workers() -> int:
    if os.name == "nt":
        return 0
    cpu = os.cpu_count() or 4
    return max(1, min(4, cpu))


def _sanitize_name(x: str) -> str:
    x = str(x or "").strip()
    if not x:
        return "dataset"
    x = re.sub(r"[^\w\-\.]+", "_", x)
    return x[:80] if len(x) > 80 else x


def _as_abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def get_args():
    p = argparse.ArgumentParser(
        description="SAGMTL for multi-step OD demand forecasting on dynamic sparse graphs"
    )

    # -------------------------------------------------------------------------
    # Data and temporal windows
    # -------------------------------------------------------------------------
    p.add_argument("--od_root", type=str, default=r"D:\pythonproject\X\OD_output_multi")
    p.add_argument("--dataset_name", type=str, default="")

    p.add_argument("--T_in", type=int, default=18)
    p.add_argument("--time_dim", type=int, default=64)
    p.add_argument("--stride_train", type=int, default=1)
    p.add_argument("--stride_eval", type=int, default=1)
    p.add_argument("--step_ahead", type=int, default=1)
    p.add_argument("--horizon", type=int, default=6)

    # -------------------------------------------------------------------------
    # Model dimensions
    # -------------------------------------------------------------------------
    p.add_argument("--hidden_dim", type=int, default=96)
    p.add_argument("--edge_dim", type=int, default=128)
    p.add_argument("--role_dim", type=int, default=96)
    p.add_argument("--edge_chunk_size", type=int, default=1024)

    # -------------------------------------------------------------------------
    # Node-edge cooperative blocks
    # -------------------------------------------------------------------------
    p.add_argument("--num_co_layers", type=int, default=2)
    p.add_argument("--co_channels", type=int, default=4)
    p.add_argument("--co_dropout", type=float, default=0.1)
    p.add_argument("--use_directional_co", type=int, choices=[0, 1], default=0)

    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------
    p.add_argument("--enc_dropout", type=float, default=0.1)

    p.add_argument("--use_edge_history", type=int, choices=[0, 1], default=1)
    p.add_argument(
        "--edge_history_input",
        type=str,
        default="flow_exist",
        choices=["flow", "flow_exist", "raw_flow", "raw_flow_exist"],
    )
    p.add_argument("--edge_history_dim", type=int, default=96)
    p.add_argument("--edge_history_chunk_size", type=int, default=4096)
    p.add_argument("--edge_history_fuse", type=str, default="concat", choices=["concat", "gate"])

    p.add_argument("--use_time_dyn", type=int, choices=[0, 1], default=1)
    p.add_argument("--use_static_node", type=int, choices=[0, 1], default=1)
    p.add_argument("--static_weight", type=float, default=0.5)
    p.add_argument("--dyn_weight", type=float, default=0.5)
    p.add_argument("--normalize_blend", type=int, choices=[0, 1], default=1)
    p.add_argument("--static_dyn_fuse_mode", type=str, default="concat", choices=["add", "concat"])

    p.add_argument("--use_time_attention", type=int, choices=[0, 1], default=0)
    p.add_argument("--time_attn_heads", type=int, default=4)
    p.add_argument("--time_attn_layers", type=int, default=1)
    p.add_argument("--time_attn_ff_mult", type=int, default=2)
    p.add_argument("--time_attn_causal", type=int, choices=[0, 1], default=0)
    p.add_argument("--time_pool_mode", type=str, default="last", choices=["last", "mean"])

    # -------------------------------------------------------------------------
    # Spatial distance-biased node attention
    # -------------------------------------------------------------------------
    p.add_argument("--use_spatial_attn", type=int, choices=[0, 1], default=1)
    p.add_argument("--spatial_dist_path", type=str, default="")
    p.add_argument("--spatial_attn_heads", type=int, default=4)
    p.add_argument("--spatial_attn_dropout", type=float, default=0.1)
    p.add_argument("--spatial_attn_res_weight", type=float, default=0.15)
    p.add_argument("--spatial_attn_exclude_self", type=int, choices=[0, 1], default=1)
    p.add_argument("--use_spatial_dist_decay", type=int, choices=[0, 1], default=1)
    p.add_argument("--spatial_dist_decay_init", type=float, default=0.1)
    p.add_argument("--spatial_dist_decay_per_head", type=int, choices=[0, 1], default=0)
    p.add_argument("--spatial_dist_norm", type=str, default="max", choices=["none", "max", "log1p", "mean"])

    # -------------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------------
    p.add_argument("--dec_dropout", type=float, default=0.1)

    p.add_argument("--use_gate1", type=int, choices=[0, 1], default=1)
    p.add_argument("--use_last_flow_gate", type=int, choices=[0, 1], default=1)
    p.add_argument("--use_sd_gate", type=int, choices=[0, 1], default=1)
    p.add_argument("--fixed_alpha", type=float, default=0.7)
    p.add_argument("--use_od_gate", type=int, choices=[0, 1], default=1)
    p.add_argument("--gate2_detach", type=int, choices=[0, 1], default=1)

    p.add_argument("--use_residual_bias", type=int, choices=[0, 1], default=1)
    p.add_argument("--use_single_head_fusion", type=int, choices=[0, 1], default=0)

    p.add_argument("--dyn_decoder_mode", type=str, default="residual", choices=["residual", "exp_clip"])
    p.add_argument("--dyn_res_use_tanh", type=int, choices=[0, 1], default=1)
    p.add_argument("--dyn_res_scale", type=float, default=1.0)
    p.add_argument("--max_log_gain", type=float, default=2.5)

    p.add_argument("--use_edge_history_in_decoder", type=int, choices=[0, 1], default=1)
    p.add_argument("--edge_history_decoder_detach", type=int, choices=[0, 1], default=0)
    p.add_argument("--static_branch_mode", type=str, default="static_roles", choices=["static_roles", "co"])
    p.add_argument(
        "--sd_fusion_mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "no_floor", "fixed", "simple_learned", "struct_dyn"],
    )

    p.add_argument("--use_static_basis_head", type=int, choices=[0, 1], default=0)
    p.add_argument("--static_basis_rank", type=int, default=4)
    p.add_argument("--static_basis_dropout", type=float, default=0.1)

    p.add_argument("--use_dyn_query_head", type=int, choices=[0, 1], default=0)
    p.add_argument("--dyn_query_heads", type=int, default=4)
    p.add_argument("--dyn_query_ff_mult", type=int, default=2)
    p.add_argument("--dyn_query_dropout", type=float, default=0.1)
    p.add_argument("--dyn_query_tokens", type=int, default=4)

    p.add_argument("--dyn_share_floor", type=float, default=0.35)
    p.add_argument("--static_scale", type=float, default=0.60)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0)

    p.add_argument("--num_workers", type=int, default=_default_num_workers())
    p.add_argument("--pin_memory", type=int, choices=[0, 1], default=1)

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gpus", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--save_dir", type=str, default="./runs")
    p.add_argument("--exp_name", type=str, default="sagmtl")
    p.add_argument("--patience", type=int, default=15)

    # -------------------------------------------------------------------------
    # Loss objective
    # -------------------------------------------------------------------------
    p.add_argument("--w_base", type=float, default=1.0)
    p.add_argument("--lambda_base", type=float, default=3.0)
    p.add_argument("--lambda_comp", type=float, default=1.0)
    p.add_argument("--comp_weight_mode", type=str, default="legacy", choices=["legacy", "direct_log", "direct_raw", "none"])
    p.add_argument("--flow_loss_space", type=str, default="log1p", choices=["log1p", "raw"])
    p.add_argument("--flow_loss_type", type=str, default="smoothl1", choices=["smoothl1", "mse"])

    p.add_argument("--w_cons", type=float, default=0.1)
    p.add_argument("--w_node", type=float, default=0.08)
    p.add_argument("--w_edge", type=float, default=0.08)
    p.add_argument("--w_vol", type=float, default=0.1)
    p.add_argument("--w_zero", type=float, default=0.035)

    p.add_argument("--edge_loss_type", type=str, default="focal", choices=["focal", "bce"])
    p.add_argument("--edge_alpha", type=float, default=0.75)
    p.add_argument("--edge_gamma", type=float, default=2.0)
    p.add_argument("--transition_boost", type=float, default=2.0)
    p.add_argument("--use_transition_weight", type=int, choices=[0, 1], default=1)

    p.add_argument("--zero_top_ratio", type=float, default=0.01)
    p.add_argument("--zero_min_topk", type=int, default=64)
    p.add_argument("--zero_max_topk", type=int, default=2048)

    p.add_argument("--aux_warmup_epochs", type=int, default=15)
    p.add_argument("--cons_warmup_epochs", type=int, default=10)
    p.add_argument("--structure_pretrain_epochs", type=int, default=5)
    p.add_argument("--structure_pretrain_base_scale", type=float, default=0.30)
    p.add_argument("--reg_warmup_epochs", type=int, default=8)
    p.add_argument("--freeze_static_epochs", type=int, default=0)

    p.add_argument("--cons_out_weight", type=float, default=1.0)
    p.add_argument("--cons_in_weight", type=float, default=1.0)

    # -------------------------------------------------------------------------
    # Evaluation and sampling
    # -------------------------------------------------------------------------
    p.add_argument("--eval_threshold", type=float, default=0.5)
    p.add_argument("--reg_masked", type=int, choices=[0, 1], default=1)
    p.add_argument("--reg_null_val", type=float, default=0.0)
    p.add_argument("--mape_eps", type=float, default=2.0)
    p.add_argument(
        "--early_stop_metric",
        type=str,
        default="composite_v1",
        choices=["mae_masked", "mae_all", "horizon_mae_masked_mean", "horizon_mae_all_mean", "composite_v1"],
    )

    p.add_argument("--edge_neg_ratio", type=float, default=1.0)
    p.add_argument("--neg_active_only", type=int, choices=[0, 1], default=1)
    p.add_argument("--train_metric_max_batches", type=int, default=8)

    args = p.parse_args()

    if args.device == "cuda":
        if args.gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
            args.local_gpu = 0
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            args.local_gpu = 0
    else:
        args.local_gpu = -1

    args.od_root = _as_abs_path(args.od_root)
    args.save_dir = _as_abs_path(args.save_dir)

    if str(args.spatial_dist_path).strip():
        args.spatial_dist_path = _as_abs_path(args.spatial_dist_path)
    elif int(args.use_spatial_attn) == 1:
        args.spatial_dist_path = os.path.join(args.od_root, "spatial_dist.npy")

    os.makedirs(args.save_dir, exist_ok=True)

    dataset_name = str(args.dataset_name).strip()
    if not dataset_name:
        dataset_name = os.path.basename(args.od_root.rstrip("/\\"))
    args.dataset_name = _sanitize_name(dataset_name)

    # Compatibility aliases for older utility code and old checkpoint metadata.
    args.w_flow = float(args.w_base)
    args.w_od = float(args.w_node)
    args.peak_alpha = float(args.lambda_comp)
    args.peak_weight_mode = str(args.comp_weight_mode)
    args.exist_loss_type = str(args.edge_loss_type)
    args.exist_alpha = float(args.edge_alpha)
    args.exist_gamma = float(args.edge_gamma)
    args.reg_lambda_cond = 0.5 * float(args.lambda_base)
    args.reg_lambda_final = 0.5 * float(args.lambda_base)
    args.lambda_base_cond = 0.5 * float(args.lambda_base)
    args.lambda_base_final = 0.5 * float(args.lambda_base)

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{_sanitize_name(args.exp_name)}_{args.dataset_name}_{ts}_pid{os.getpid()}"
    args.run_name = run_name
    args.exp_dir = os.path.join(args.save_dir, args.dataset_name, run_name)
    os.makedirs(args.exp_dir, exist_ok=True)

    with open(os.path.join(args.exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    return args
