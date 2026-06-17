# utils/data_utils.py
# -*- coding: utf-8 -*-
import glob
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


BIN_MINUTES = 20
BIN_FREQ = f"{BIN_MINUTES}min"
DEFAULT_STEP_AHEAD = 1


def build_node_dynamic_features_raw(edges_df: pd.DataFrame, num_nodes: int) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Build raw node-level dynamic features from OD edges.

    Feature order:
        0: out_degree
        1: in_degree
        2: out_flow
        3: in_flow
    """
    if edges_df.empty:
        raise ValueError("[data] Empty OD edge table; cannot build node dynamic features.")

    edges = edges_df.copy()
    edges["dt"] = pd.to_datetime(edges["dt"], errors="coerce")
    edges = edges.dropna(subset=["dt"])
    edges["dt"] = edges["dt"].dt.floor(BIN_FREQ)

    edges["dt_ns"] = edges["dt"].astype("int64")
    times_ns = np.sort(edges["dt_ns"].unique())
    if times_ns.size == 0:
        raise ValueError("[data] No valid timestamp remains after cleaning OD edges.")

    dyn_by_ns = {int(ns): np.zeros((num_nodes, 4), dtype=np.float32) for ns in times_ns}

    for dt_ns, group in edges.groupby("dt_ns"):
        dyn = dyn_by_ns[int(dt_ns)]
        src = group["src"].to_numpy(dtype=np.int64)
        dst = group["dst"].to_numpy(dtype=np.int64)
        flow = group["flow"].to_numpy(dtype=np.float32)

        src_mask = (src >= 0) & (src < num_nodes)
        if src_mask.any():
            src_valid = src[src_mask]
            flow_valid = flow[src_mask]
            np.add.at(dyn[:, 0], src_valid, 1.0)
            np.add.at(dyn[:, 2], src_valid, flow_valid)

        dst_mask = (dst >= 0) & (dst < num_nodes)
        if dst_mask.any():
            dst_valid = dst[dst_mask]
            flow_valid = flow[dst_mask]
            np.add.at(dyn[:, 1], dst_valid, 1.0)
            np.add.at(dyn[:, 3], dst_valid, flow_valid)

    return {
        pd.Timestamp(int(ns)).tz_localize(None): dyn_by_ns[int(ns)]
        for ns in times_ns
    }


def fourier_time_features(times: List[pd.Timestamp], time_dim: int = 64) -> torch.Tensor:
    """Return Fourier time features with daily and weekly periodic terms."""
    T = len(times)
    time_dim = int(time_dim) if time_dim and time_dim > 0 else 64
    k = max(1, time_dim // 4)

    hour_of_day = np.array([t.hour + t.minute / 60.0 for t in times], dtype=np.float32)
    day_of_week = np.array([t.weekday() for t in times], dtype=np.float32)

    feats = []
    for j in range(1, k + 1):
        w = 2.0 * np.pi * j / 24.0
        feats.append(np.sin(w * hour_of_day))
        feats.append(np.cos(w * hour_of_day))
    for j in range(1, k + 1):
        w = 2.0 * np.pi * j / 7.0
        feats.append(np.sin(w * day_of_week))
        feats.append(np.cos(w * day_of_week))

    X = np.stack(feats, axis=1).astype(np.float32)
    if X.shape[1] < time_dim:
        pad = np.zeros((T, time_dim - X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, pad], axis=1)
    elif X.shape[1] > time_dim:
        X = X[:, :time_dim]
    return torch.from_numpy(X)


def _single_collate(batch):
    return batch[0]


def _glob_many(root: str, patterns: List[str]) -> List[str]:
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(root, pattern), recursive=True))
    return sorted(set(p for p in paths if os.path.isfile(p)))


def _list_all_files(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return sorted(files)


def _normalize_edge_csv(df: pd.DataFrame, src_path: str) -> pd.DataFrame:
    """Normalize edge CSV columns to dt/src/dst/flow."""
    if df.empty:
        return df

    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ("dt", "time", "timestamp", "hour"):
            rename_map[col] = "dt"
        elif col_lower in ("src", "o", "origin", "from"):
            rename_map[col] = "src"
        elif col_lower in ("dst", "d", "dest", "to"):
            rename_map[col] = "dst"
        elif col_lower in ("flow", "cnt", "weight", "y"):
            rename_map[col] = "flow"

    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"dt", "src", "dst", "flow"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[data] {os.path.basename(src_path)} is missing columns {missing}; "
            f"available columns={list(df.columns)}"
        )

    out = df.assign(
        dt=pd.to_datetime(df["dt"], errors="coerce").dt.floor(BIN_FREQ),
        src=pd.to_numeric(df["src"], errors="coerce").astype("Int64"),
        dst=pd.to_numeric(df["dst"], errors="coerce").astype("Int64"),
        flow=pd.to_numeric(df["flow"], errors="coerce").astype(np.float32),
    ).dropna(subset=["dt", "src", "dst", "flow"])

    out["src"] = out["src"].astype(np.int64)
    out["dst"] = out["dst"].astype(np.int64)
    return out[["dt", "src", "dst", "flow"]].copy()


@dataclass
class HourCSVStore:
    """
    CSV-backed OD store with a fixed global candidate-edge space.

    The store creates:
        - a continuous 20-minute timeline,
        - a fixed global candidate edge set from historical OD edges,
        - per-slot sparse edge caches,
        - train-fitted node dynamic normalization.
    """

    root: str
    node_feat_candidates: tuple = ("node_features.npy", "grid_node_features.npy", "features.npy")
    edges_candidates: tuple = (
        "od_edges_*.csv",
        "edges_hour.csv",
        "edge_hour.csv",
        "od_edges_hour.csv",
        "edges*.csv",
        "edges_hour/*.csv",
        "**/edges_hour*.csv",
        "**/edges_*.csv",
    )

    def __post_init__(self):
        self.root = os.path.abspath(self.root)

        feat_path = self._find_node_feature_file()
        node_feat = np.load(feat_path).astype(np.float32)
        if node_feat.ndim != 2:
            raise ValueError(f"[data] Node features must be 2D, got shape={node_feat.shape}")
        self.N = int(node_feat.shape[0])
        self.node_feat = torch.from_numpy(node_feat)

        df_e = self._load_edge_dataframe()
        self._expand_nodes_if_needed(df_e)
        self._build_time_axis(df_e)

        df_e = df_e.copy()
        df_e["dt_ns"] = df_e["dt"].astype("int64")
        df_e = df_e.groupby(["dt_ns", "src", "dst"], as_index=False)["flow"].sum()

        self._build_global_edge_space(df_e)
        self._build_sparse_edge_cache(df_e)
        self._build_node_dynamic_cache(df_e)

    def _find_node_feature_file(self) -> str:
        for name in self.node_feat_candidates:
            path = os.path.join(self.root, name)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"[data] Missing node feature file: {self.node_feat_candidates}")

    def _load_edge_dataframe(self) -> pd.DataFrame:
        edge_paths = _glob_many(self.root, list(self.edges_candidates))
        if not edge_paths:
            existing = _list_all_files(self.root)
            hint = "\n  - " + "\n  - ".join(existing[:30])
            if len(existing) > 30:
                hint += "\n  ..."
            raise FileNotFoundError(
                "[data] Missing OD edge CSV files. Expected one of these patterns:\n"
                f"  {self.edges_candidates}\n"
                f"[data] Example files under root:{hint}"
            )

        frames = []
        for path in edge_paths:
            df = pd.read_csv(path)
            df = _normalize_edge_csv(df, path)
            if not df.empty:
                frames.append(df)

        if not frames:
            raise ValueError("[data] All matched OD edge CSV files are empty.")

        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    def _expand_nodes_if_needed(self, df_e: pd.DataFrame):
        max_node_idx = int(max(df_e["src"].max(), df_e["dst"].max())) if not df_e.empty else self.N - 1
        target_N = max(self.N, max_node_idx + 1)
        if target_N <= self.N:
            return

        F = int(self.node_feat.shape[1])
        pad = torch.zeros((target_N - self.N, F), dtype=self.node_feat.dtype)
        self.node_feat = torch.cat([self.node_feat, pad], dim=0)
        self.N = target_N

    def _build_time_axis(self, df_e: pd.DataFrame):
        dt_min = pd.to_datetime(df_e["dt"].min()).floor(BIN_FREQ)
        dt_max = pd.to_datetime(df_e["dt"].max()).floor(BIN_FREQ)
        if pd.isna(dt_min) or pd.isna(dt_max):
            raise ValueError("[data] Failed to parse a valid time range from edge CSV files.")
        if dt_max < dt_min:
            raise ValueError(f"[data] Invalid time range: dt_min={dt_min}, dt_max={dt_max}")

        full_times = pd.date_range(start=dt_min, end=dt_max, freq=BIN_FREQ)
        self.times = [pd.Timestamp(t).tz_localize(None) for t in full_times]
        self.T_total = len(self.times)
        self._dt_to_idx: Dict[int, int] = {
            int(pd.Timestamp(t).value): i for i, t in enumerate(self.times)
        }

    def _build_global_edge_space(self, df_e: pd.DataFrame):
        pairs = sorted(set(zip(df_e["src"].tolist(), df_e["dst"].tolist())))
        if not pairs:
            self.edge_index_all_global = np.zeros((2, 0), dtype=np.int64)
            self.edge_idx_map: Dict[Tuple[int, int], int] = {}
        else:
            self.edge_index_all_global = np.array(pairs, dtype=np.int64).T
            self.edge_idx_map = {(int(s), int(d)): i for i, (s, d) in enumerate(pairs)}

        self.E_global = int(self.edge_index_all_global.shape[1])
        self.edge_index_all_global_torch = torch.from_numpy(self.edge_index_all_global.copy())

    def _build_sparse_edge_cache(self, df_e: pd.DataFrame):
        self._edges_by_t: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for dt_ns, sub in df_e.groupby("dt_ns"):
            time_idx = self._dt_to_idx.get(int(dt_ns), None)
            if time_idx is None:
                continue
            src = sub["src"].to_numpy(dtype=np.int64)
            dst = sub["dst"].to_numpy(dtype=np.int64)
            flow = sub["flow"].to_numpy(dtype=np.float32)
            self._edges_by_t[time_idx] = (src, dst, flow)

    def _build_node_dynamic_cache(self, df_e: pd.DataFrame):
        if self.T_total <= 0:
            self._node_dyn_log_all = None
            self.node_dyn_all = torch.zeros((0, self.N, 4), dtype=torch.float32)
            self.node_dyn_meta = {"transform": "none"}
            return

        df_dyn = df_e.copy()
        df_dyn["dt"] = pd.to_datetime(df_dyn["dt_ns"], unit="ns")
        edges_dyn = df_dyn[["dt", "src", "dst", "flow"]].copy()
        dyn_by_dt_raw = build_node_dynamic_features_raw(edges_dyn, num_nodes=self.N)

        dyn_dim = 4
        dyn_raw_all = np.zeros((self.T_total, self.N, dyn_dim), dtype=np.float32)
        for i, t in enumerate(self.times):
            feat = dyn_by_dt_raw.get(pd.to_datetime(t))
            if feat is None:
                continue
            if feat.shape[0] < self.N:
                pad = np.zeros((self.N - feat.shape[0], feat.shape[1]), dtype=np.float32)
                feat = np.concatenate([feat, pad], axis=0)
            elif feat.shape[0] > self.N:
                feat = feat[: self.N]
            dyn_raw_all[i] = feat.astype(np.float32)

        self._node_dyn_log_all = np.log1p(dyn_raw_all)
        self.node_dyn_all = None
        self.node_dyn_meta = {"transform": "log1p_z(train-fit)", "mean": None, "std": None}

    def fit_node_dyn_norm(self, train_time_indices: np.ndarray):
        """Fit node dynamic feature normalization on train time indices only."""
        if self._node_dyn_log_all is None:
            return

        idx = np.unique(train_time_indices.astype(np.int64))
        idx = idx[(idx >= 0) & (idx < self.T_total)]
        X_fit = self._node_dyn_log_all if idx.size == 0 else self._node_dyn_log_all[idx]

        mean = X_fit.mean(axis=(0, 1), keepdims=True)
        std = X_fit.std(axis=(0, 1), keepdims=True)
        std_safe = np.where(std < 1e-6, 1.0, std)

        X_all = (self._node_dyn_log_all - mean) / std_safe
        self.node_dyn_all = torch.from_numpy(X_all.astype(np.float32))
        self.node_dyn_meta = {
            "transform": "log1p_z(train-fit)",
            "mean": mean.reshape(-1).tolist(),
            "std": std.reshape(-1).tolist(),
        }

    def _load_edges_at(self, time_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._edges_by_t.get(
            int(time_idx),
            (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)),
        )

    def _build_hist_edge_index(self, t_hist: List[int]) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
        """Backward-compatible helper. The current pipeline uses the fixed global edge space."""
        hist_pairs = set()
        for time_idx in t_hist:
            src_i, dst_i, _ = self._load_edges_at(time_idx)
            if src_i.size > 0:
                hist_pairs.update(zip(src_i.tolist(), dst_i.tolist()))

        if not hist_pairs:
            return np.zeros((2, 0), dtype=np.int64), {}

        pairs = sorted(hist_pairs)
        edge_index_np = np.array(pairs, dtype=np.int64).T
        idx_map = {(int(s), int(d)): i for i, (s, d) in enumerate(pairs)}
        return edge_index_np, idx_map

    def get_window(
        self,
        start_idx: int,
        T_in: int,
        time_dim: int,
        use_last_flow_gate: bool = True,
        edge_neg_ratio: float = 1.0,
        neg_active_only: bool = True,
        step_ahead: int = DEFAULT_STEP_AHEAD,
        horizon: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Build one fixed-edge, multi-horizon OD forecasting window."""
        del edge_neg_ratio, neg_active_only

        step_ahead = int(step_ahead)
        horizon = int(horizon)
        if step_ahead <= 0:
            raise ValueError(f"[data] step_ahead must be positive, got {step_ahead}")
        if horizon <= 0:
            raise ValueError(f"[data] horizon must be positive, got {horizon}")

        t_hist = list(range(int(start_idx), int(start_idx) + int(T_in)))
        t_targets = [t_hist[-1] + step_ahead + h for h in range(horizon)]

        N = self.N
        E = self.E_global

        node_time_feat = self._build_node_time_feat(t_hist=t_hist, N=N, time_dim=time_dim)
        edge_index = self.edge_index_all_global_torch

        edge_labels_seq, y_od_seq, node_od_label_seq, target_dt_seq = self._build_targets(
            t_targets=t_targets,
            horizon=horizon,
            E=E,
            N=N,
        )

        edge_labels = edge_labels_seq[0].clone()
        y_od = y_od_seq[0].clone()
        node_od_label = node_od_label_seq[0].clone()

        last_flow = self._build_last_flow(t_hist=t_hist, E=E, enabled=use_last_flow_gate)
        edge_flow_hist = self._build_edge_flow_history(t_hist=t_hist, E=E)
        edge_exist_hist = (edge_flow_hist > 0).float()

        hist_last_dt = self.times[t_hist[-1]]
        target_dt = target_dt_seq[0]
        delta_min = int((pd.Timestamp(target_dt) - pd.Timestamp(hist_last_dt)).total_seconds() // 60)
        expected_min = int(step_ahead * BIN_MINUTES)

        return {
            "node_feat": self.node_feat,
            "node_time_feat": node_time_feat,
            "edge_index": edge_index,
            "edge_index_all": edge_index,

            "edge_labels": edge_labels,
            "y_od": y_od,
            "node_od_label": node_od_label,

            "edge_labels_seq": edge_labels_seq,
            "y_od_seq": y_od_seq,
            "node_od_label_seq": node_od_label_seq,

            "last_flow": last_flow,
            "edge_flow_hist": edge_flow_hist,
            "edge_exist_hist": edge_exist_hist,

            "target_dt": target_dt,
            "target_dt_seq": target_dt_seq,
            "hist_dt_seq": [self.times[i] for i in t_hist],

            "sample_id": int(start_idx),
            "t_hist_idx_seq": [int(x) for x in t_hist],
            "t_target_idx": int(t_targets[0]),
            "t_target_idx_seq": [int(x) for x in t_targets],
            "t_hist_last_idx": int(t_hist[-1]),

            "edge_src": edge_index[0].clone(),
            "edge_dst": edge_index[1].clone(),

            "step_ahead": int(step_ahead),
            "horizon": int(horizon),
            "delta_expected": expected_min,

            "hist_last_dt": hist_last_dt,
            "delta_minutes": delta_min,
            "delta_ok": int(delta_min == expected_min),
        }

    def _build_node_time_feat(self, t_hist: List[int], N: int, time_dim: int) -> torch.Tensor:
        times_hist = [self.times[i] for i in t_hist]
        dyn_dim = 4
        base_time_dim = max(int(time_dim) - dyn_dim, 1)

        time_feat = fourier_time_features(times_hist, time_dim=base_time_dim)
        node_time_fourier = time_feat.unsqueeze(0).repeat(N, 1, 1).contiguous()

        if (
            getattr(self, "node_dyn_all", None) is not None
            and isinstance(self.node_dyn_all, torch.Tensor)
            and self.node_dyn_all.numel() > 0
        ):
            dyn_seq = self.node_dyn_all[t_hist]
            dyn_seq = dyn_seq[:, :N, :dyn_dim].permute(1, 0, 2).contiguous()
        else:
            dyn_seq = torch.zeros((N, len(t_hist), dyn_dim), dtype=torch.float32)

        node_time_feat = torch.cat([node_time_fourier, dyn_seq], dim=-1)
        if node_time_feat.size(-1) < time_dim:
            pad = torch.zeros(
                N,
                len(t_hist),
                int(time_dim) - node_time_feat.size(-1),
                dtype=node_time_feat.dtype,
            )
            node_time_feat = torch.cat([node_time_feat, pad], dim=-1)
        elif node_time_feat.size(-1) > time_dim:
            node_time_feat = node_time_feat[..., :time_dim]
        return node_time_feat

    def _build_targets(
        self,
        t_targets: List[int],
        horizon: int,
        E: int,
        N: int,
    ):
        edge_labels_seq_np = np.zeros((horizon, E), dtype=np.float32)
        y_od_seq_np = np.zeros((horizon, E), dtype=np.float32)
        node_od_label_seq_np = np.zeros((horizon, N, 2), dtype=np.float32)
        target_dt_seq = []

        for h, t_target in enumerate(t_targets):
            src_t, dst_t, flow_t = self._load_edges_at(t_target)
            target_dt_seq.append(self.times[t_target])

            if src_t.size > 0 and E > 0:
                for s, d, f in zip(src_t.tolist(), dst_t.tolist(), flow_t.tolist()):
                    edge_idx = self.edge_idx_map.get((int(s), int(d)), None)
                    if edge_idx is not None:
                        edge_labels_seq_np[h, edge_idx] = float(f)
                        y_od_seq_np[h, edge_idx] = 1.0

            if src_t.size > 0:
                out_flow = np.bincount(src_t, weights=flow_t, minlength=N).astype(np.float32)
                in_flow = np.bincount(dst_t, weights=flow_t, minlength=N).astype(np.float32)
            else:
                out_flow = np.zeros((N,), dtype=np.float32)
                in_flow = np.zeros((N,), dtype=np.float32)

            node_od_label_seq_np[h, :, 0] = (out_flow > 0).astype(np.float32)
            node_od_label_seq_np[h, :, 1] = (in_flow > 0).astype(np.float32)

        return (
            torch.from_numpy(edge_labels_seq_np),
            torch.from_numpy(y_od_seq_np),
            torch.from_numpy(node_od_label_seq_np),
            target_dt_seq,
        )

    def _build_last_flow(self, t_hist: List[int], E: int, enabled: bool) -> torch.Tensor:
        if (not enabled) or E <= 0:
            return torch.zeros((E,), dtype=torch.float32)

        src_prev, dst_prev, flow_prev = self._load_edges_at(t_hist[-1])
        last_flow_np = np.zeros((E,), dtype=np.float32)
        if src_prev.size > 0:
            for s, d, f in zip(src_prev.tolist(), dst_prev.tolist(), flow_prev.tolist()):
                edge_idx = self.edge_idx_map.get((int(s), int(d)), None)
                if edge_idx is not None:
                    last_flow_np[edge_idx] = float(f)
        return torch.from_numpy(last_flow_np)

    def _build_edge_flow_history(self, t_hist: List[int], E: int) -> torch.Tensor:
        edge_flow_hist_np = np.zeros((E, len(t_hist)), dtype=np.float32)
        if E <= 0:
            return torch.from_numpy(edge_flow_hist_np)

        for tau, time_idx in enumerate(t_hist):
            src_h, dst_h, flow_h = self._load_edges_at(time_idx)
            if src_h.size == 0:
                continue
            for s, d, f in zip(src_h.tolist(), dst_h.tolist(), flow_h.tolist()):
                edge_idx = self.edge_idx_map.get((int(s), int(d)), None)
                if edge_idx is not None:
                    edge_flow_hist_np[edge_idx, tau] = float(f)
        return torch.from_numpy(edge_flow_hist_np)


class HourODDataset(Dataset):
    def __init__(
        self,
        od_root: str,
        split: str,
        T_in: int = 18,
        stride: int = 1,
        time_dim: int = 64,
        step_ahead: int = DEFAULT_STEP_AHEAD,
        horizon: int = 6,
        split_ratio=(0.7, 0.15, 0.15),
        use_last_flow_gate: bool = True,
        edge_neg_ratio: float = 1.0,
        neg_active_only: bool = True,
        store: Optional[HourCSVStore] = None,
    ):
        super().__init__()
        od_root = os.path.abspath(od_root)
        self.store = store if store is not None else HourCSVStore(od_root)

        self.T_in = int(T_in)
        self.time_dim = int(time_dim)
        self.stride = int(stride)
        self.step_ahead = int(step_ahead)
        self.horizon = int(horizon)

        if self.step_ahead <= 0:
            raise ValueError(f"[data] step_ahead must be positive, got {self.step_ahead}")
        if self.horizon <= 0:
            raise ValueError(f"[data] horizon must be positive, got {self.horizon}")

        self.use_last_flow_gate = bool(use_last_flow_gate)
        self.edge_neg_ratio = float(edge_neg_ratio)
        self.neg_active_only = bool(neg_active_only)

        extra = self.step_ahead + self.horizon - 1
        max_start = self.store.T_total - self.T_in - extra
        if max_start < 0:
            need = self.T_in + extra + 1
            raise ValueError(
                f"[data] T_total={self.store.T_total} is smaller than required window length={need} "
                f"(T_in={self.T_in}, step_ahead={self.step_ahead}, horizon={self.horizon})."
            )

        all_starts = np.arange(0, max_start + 1, self.stride, dtype=np.int64)
        train_ratio, val_ratio, _ = split_ratio
        n = len(all_starts)
        i1 = int(n * train_ratio)
        i2 = int(n * (train_ratio + val_ratio))

        if split == "train":
            self.starts = all_starts[:i1]
        elif split == "val":
            self.starts = all_starts[i1:i2]
        elif split == "test":
            self.starts = all_starts[i2:]
        else:
            raise ValueError("split must be one of 'train', 'val', or 'test'.")

        if split == "train" and getattr(self.store, "node_dyn_all", None) is None:
            train_time_indices = []
            for start in self.starts.tolist():
                train_time_indices.extend(range(int(start), int(start) + self.T_in))
            self.store.fit_node_dyn_norm(np.asarray(train_time_indices, dtype=np.int64))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])
        return self.store.get_window(
            start_idx=start,
            T_in=self.T_in,
            time_dim=self.time_dim,
            use_last_flow_gate=self.use_last_flow_gate,
            edge_neg_ratio=self.edge_neg_ratio,
            neg_active_only=self.neg_active_only,
            step_ahead=self.step_ahead,
            horizon=self.horizon,
        )


def _move_to_device(x: Any, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_move_to_device(v, device) for v in x)
    return x


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return _move_to_device(batch, device)


def _build_loader_kwargs(
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_single_collate,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def get_hour_od_dataloaders(
    od_root: str,
    T_in: int,
    time_dim: int,
    step_ahead: int = DEFAULT_STEP_AHEAD,
    horizon: int = 6,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    stride_train: int = 1,
    stride_eval: int = 1,
    use_last_flow_gate: bool = True,
    split_ratio=(0.7, 0.15, 0.15),
    edge_neg_ratio: float = 1.0,
    neg_active_only: bool = True,
):
    step_ahead = int(step_ahead)
    horizon = int(horizon)

    shared_store = HourCSVStore(od_root)

    ds_tr = HourODDataset(
        od_root=od_root,
        split="train",
        T_in=T_in,
        stride=stride_train,
        time_dim=time_dim,
        step_ahead=step_ahead,
        horizon=horizon,
        split_ratio=split_ratio,
        use_last_flow_gate=use_last_flow_gate,
        edge_neg_ratio=edge_neg_ratio,
        neg_active_only=neg_active_only,
        store=shared_store,
    )
    ds_va = HourODDataset(
        od_root=od_root,
        split="val",
        T_in=T_in,
        stride=stride_eval,
        time_dim=time_dim,
        step_ahead=step_ahead,
        horizon=horizon,
        split_ratio=split_ratio,
        use_last_flow_gate=use_last_flow_gate,
        edge_neg_ratio=edge_neg_ratio,
        neg_active_only=neg_active_only,
        store=shared_store,
    )
    ds_te = HourODDataset(
        od_root=od_root,
        split="test",
        T_in=T_in,
        stride=stride_eval,
        time_dim=time_dim,
        step_ahead=step_ahead,
        horizon=horizon,
        split_ratio=split_ratio,
        use_last_flow_gate=use_last_flow_gate,
        edge_neg_ratio=edge_neg_ratio,
        neg_active_only=neg_active_only,
        store=shared_store,
    )

    if batch_size != 1:
        warnings.warn(
            "[data] Current training pipeline expects batch_size=1 for a full fixed-edge graph; "
            "forcing batch_size=1.",
            stacklevel=1,
        )
    batch_size = 1

    dl_tr = DataLoader(
        ds_tr,
        **_build_loader_kwargs(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )
    dl_va = DataLoader(
        ds_va,
        **_build_loader_kwargs(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )
    dl_te = DataLoader(
        ds_te,
        **_build_loader_kwargs(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )

    pred_start_minutes = step_ahead * BIN_MINUTES
    pred_end_minutes = (step_ahead + horizon - 1) * BIN_MINUTES

    if horizon == 1:
        predict_target = f"t+{step_ahead}"
        predict_horizon = predict_target
    else:
        predict_target = f"t+{step_ahead}"
        predict_horizon = f"t+{step_ahead} ... t+{step_ahead + horizon - 1}"

    meta = {
        "N": shared_store.N,
        "F": int(shared_store.node_feat.shape[1]),
        "T_total": shared_store.T_total,
        "train_size": len(ds_tr),
        "val_size": len(ds_va),
        "test_size": len(ds_te),
        "step_ahead": int(step_ahead),
        "horizon": int(horizon),
        "pred_minutes": pred_start_minutes,
        "pred_minutes_end": pred_end_minutes,
        "predict_target": predict_target,
        "predict_horizon": predict_horizon,
        "edge_neg_ratio": edge_neg_ratio,
        "neg_active_only": neg_active_only,
        "E_all": int(shared_store.edge_index_all_global.shape[1]),
        "bin_minutes": BIN_MINUTES,
        "node_dyn_transform": shared_store.node_dyn_meta.get("transform", "unknown"),
        "edge_space": "fixed_global",
    }
    return {"train": dl_tr, "val": dl_va, "test": dl_te}, meta
