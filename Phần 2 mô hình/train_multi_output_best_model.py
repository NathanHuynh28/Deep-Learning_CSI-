from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from multi_output_attention_gru import MultiOutputAttentionGRU

ROOT_DIR = Path(__file__).resolve().parents[1]
PHASE2_DIR = Path(__file__).resolve().parent
PART1_DIR = ROOT_DIR / "Phần 1 tiền xử lí"
NORMALIZE_PATH = PART1_DIR / "Normalization" / "CSI_normalize.py"
WINDOWS_DIR = PART1_DIR / "processed" / "windows"
ATTENTION_RUNS_DIR = PHASE2_DIR / "runs" / "attention_gru"
RUNS_DIR = PHASE2_DIR / "runs" / "multi_output_best_model"
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
PROFILES = {"raw", "raw_shift", "phase_hampel", "phase_hampel_shift"}
SPLITS = ("train", "val", "test")
CENTER_MAX_M = 2.7

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
Target = dict[str, torch.Tensor]
Batch = tuple[torch.Tensor, Target]
Row = dict[str, object]


class NormalizeModule(Protocol):
    def load_stats(self, profile_dir: Path) -> dict[str, FloatArray]: ...
    def apply_normalization(self, x: np.ndarray, stats: dict[str, FloatArray]) -> FloatArray: ...


def load_normalize_module() -> NormalizeModule:
    spec = importlib.util.spec_from_file_location("csi_normalize_multi_output", NORMALIZE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CSI_normalize.py from {NORMALIZE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(NormalizeModule, cast(object, module))


normalize_module = load_normalize_module()


@dataclass
class TrainConfig:
    phase_name: str = "Multi-output Attention-GRU"
    profile: str = "raw_shift"
    hidden_size: int = 96
    attention_dim: int = 32
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_layers: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    cell_head_type: str = "linear"
    cell_head_hidden_size: int | None = None
    cell_head_dropout: float = 0.0
    num_workers: int = 0
    normalize_runtime: bool = True
    noise_sigma: float = 0.0
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1e-8
    scheduler_factor: float = 0.5
    scheduler_patience: int | None = None
    scheduler_min_lr: float = 0.0
    seed: int = 42
    experiment_name: str = "multi_output_best_model"
    device: str = "auto"
    disable_progress: bool = False
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    loss_weight_presence: float = 1.0
    loss_weight_cell: float = 1.0
    loss_weight_pose: float = 1.0
    loss_weight_center: float = 1.0
    use_presence_pos_weight: bool = True
    cell_loss_type: str = "cross_entropy"
    cell_label_smoothing: float = 0.0
    cell_focal_gamma: float = 0.0
    cell_class_weighting: str = "inverse_frequency"
    cell_class_balance_beta: float = 0.9999
    selector_run_prefixes: tuple[str, ...] = ("A", "S2_A")


def validate_safe_name(name: str, field_name: str) -> str:
    path = Path(name)
    if not name or name in {".", ".."} or ".." in name or path.is_absolute() or len(path.parts) != 1:
        raise ValueError(f"{field_name} must be a safe slug, got {name!r}")
    if not SAFE_NAME_RE.fullmatch(name):
        raise ValueError(f"{field_name} may contain only ASCII letters, digits, '_', '-', and '.', got {name!r}")
    return name


def resolve_run_dir(experiment_name: str) -> Path:
    safe_name = validate_safe_name(experiment_name, "experiment_name")
    runs_root = RUNS_DIR.resolve()
    run_dir = (RUNS_DIR / safe_name).resolve()
    try:
        _ = run_dir.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError(f"experiment_name escapes managed runs directory: {experiment_name!r}") from exc
    return run_dir


def validate_config(config: TrainConfig) -> None:
    if config.phase_name != "Multi-output Attention-GRU":
        raise ValueError("phase_name must be 'Multi-output Attention-GRU'")
    _ = validate_safe_name(config.experiment_name, "experiment_name")
    _ = validate_safe_name(config.profile, "profile")
    if config.profile not in PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(sorted(PROFILES))}")
    positive_ints = {"epochs": config.epochs, "batch_size": config.batch_size, "patience": config.patience, "hidden_size": config.hidden_size, "attention_dim": config.attention_dim, "num_layers": config.num_layers}
    for name, value in positive_ints.items():
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must satisfy 0 <= dropout < 1")
    if config.cell_head_type not in {"linear", "mlp"}:
        raise ValueError("cell_head_type must be 'linear' or 'mlp'")
    if config.cell_head_hidden_size is not None and config.cell_head_hidden_size < 1:
        raise ValueError("cell_head_hidden_size must be None or >= 1")
    if not 0.0 <= config.cell_head_dropout < 1.0:
        raise ValueError("cell_head_dropout must satisfy 0 <= dropout < 1")
    for name in ("learning_rate", "grad_clip"):
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be > 0")
    for name in ("weight_decay", "noise_sigma", "loss_weight_presence", "loss_weight_cell", "loss_weight_pose", "loss_weight_center"):
        if getattr(config, name) < 0.0:
            raise ValueError(f"{name} must be >= 0")
    if config.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if not 0.0 <= config.optimizer_beta1 < 1.0:
        raise ValueError("optimizer_beta1 must satisfy 0 <= beta < 1")
    if not 0.0 <= config.optimizer_beta2 < 1.0:
        raise ValueError("optimizer_beta2 must satisfy 0 <= beta < 1")
    if config.optimizer_eps <= 0.0:
        raise ValueError("optimizer_eps must be > 0")
    if not 0.0 < config.scheduler_factor < 1.0:
        raise ValueError("scheduler_factor must satisfy 0 < factor < 1")
    if config.scheduler_patience is not None and config.scheduler_patience < 0:
        raise ValueError("scheduler_patience must be None or >= 0")
    if config.scheduler_min_lr < 0.0:
        raise ValueError("scheduler_min_lr must be >= 0")
    if config.cell_loss_type not in {"cross_entropy", "focal"}:
        raise ValueError("cell_loss_type must be 'cross_entropy' or 'focal'")
    if not 0.0 <= config.cell_label_smoothing < 1.0:
        raise ValueError("cell_label_smoothing must satisfy 0 <= smoothing < 1")
    if config.cell_focal_gamma < 0.0:
        raise ValueError("cell_focal_gamma must be >= 0")
    if config.cell_class_weighting not in {"none", "inverse_frequency", "effective_number"}:
        raise ValueError("cell_class_weighting must be none, inverse_frequency, or effective_number")
    if not 0.0 <= config.cell_class_balance_beta < 1.0:
        raise ValueError("cell_class_balance_beta must satisfy 0 <= beta < 1")
    if not config.selector_run_prefixes:
        raise ValueError("selector_run_prefixes must contain at least one planned run prefix")
    if config.max_train_batches is not None and config.max_train_batches < 1:
        raise ValueError("max_train_batches must be None or >= 1")
    if config.max_eval_batches is not None and config.max_eval_batches < 1:
        raise ValueError("max_eval_batches must be None or >= 1")


class MultiOutputCSIDataset(Dataset[Batch]):
    def __init__(self, windows_dir: Path, profile: str, split: str, normalize_runtime: bool = True, noise_sigma: float = 0.0) -> None:
        if split not in SPLITS:
            raise ValueError("split must be train, val, or test")
        if noise_sigma < 0.0:
            raise ValueError("noise_sigma must be non-negative")
        self.profile_dir: Path = windows_dir / validate_safe_name(profile, "profile")
        self.split: str = split
        self.noise_sigma: float = float(noise_sigma)
        data_path = self.profile_dir / "data.npz"
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing profile data: {data_path}")
        with cast(NpzFile, np.load(data_path)) as data:
            self.x: FloatArray = cast(FloatArray, data[f"X_{split}"].astype(np.float32, copy=False))
            self.presence: IntArray = cast(IntArray, data[f"y_presence_{split}"].astype(np.int64, copy=False))
            self.cell: IntArray = cast(IntArray, data[f"y_cell_{split}"].astype(np.int64, copy=False))
            self.pose: IntArray = cast(IntArray, data[f"y_pose_{split}"].astype(np.int64, copy=False))
            self.center: FloatArray = cast(FloatArray, data[f"y_center_{split}"].astype(np.float32, copy=False))
        self.stats: dict[str, FloatArray] | None = normalize_module.load_stats(self.profile_dir) if normalize_runtime else None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Batch:
        x = cast(FloatArray, self.x[index].astype(np.float32, copy=True))
        if self.stats is not None:
            x = cast(FloatArray, normalize_module.apply_normalization(x[np.newaxis, ...], self.stats)[0])
        if self.split == "train" and self.noise_sigma > 0.0:
            noise = cast(FloatArray, np.random.normal(0.0, self.noise_sigma, size=x.shape).astype(np.float32))
            x = cast(FloatArray, (x + noise).astype(np.float32))
        x = cast(FloatArray, x.transpose(2, 0, 1).reshape(192, 192))
        raw_cell = int(self.cell[index])
        presence = int(self.presence[index])
        human_cell = raw_cell if raw_cell < 25 else -1
        center_m = cast(FloatArray, self.center[index].astype(np.float32, copy=True))
        center_norm = cast(FloatArray, np.clip(center_m / np.float32(CENTER_MAX_M), 0.0, 1.0).astype(np.float32))
        cell_mask = bool(presence == 1 and 0 <= raw_cell < 25)
        center_mask = bool(presence == 1)
        target = {
            "presence": torch.tensor(float(presence), dtype=torch.float32),
            "raw_cell": torch.tensor(raw_cell, dtype=torch.long),
            "human_cell": torch.tensor(human_cell, dtype=torch.long),
            "cell_mask": torch.tensor(cell_mask, dtype=torch.bool),
            "pose": torch.tensor(int(self.pose[index]), dtype=torch.long),
            "center_m": torch.as_tensor(center_m, dtype=torch.float32),
            "center_norm": torch.as_tensor(center_norm, dtype=torch.float32),
            "center_mask": torch.tensor(center_mask, dtype=torch.bool),
        }
        return torch.from_numpy(x), target


def make_dataloaders(config: TrainConfig) -> dict[str, DataLoader[Batch]]:
    return {
        split: DataLoader(
            MultiOutputCSIDataset(WINDOWS_DIR, config.profile, split, config.normalize_runtime, config.noise_sigma),
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
        )
        for split in SPLITS
    }


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    _ = torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def limited_iter(loader: DataLoader[Batch], max_batches: int | None) -> Iterator[Batch]:
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield cast(Batch, batch)


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0.0 else 0.0


def class_weights_from_labels(labels: IntArray, num_classes: int) -> torch.Tensor:
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError(f"class labels must be in [0, {num_classes - 1}]")
    counts = np.bincount(labels.astype(np.int64, copy=False), minlength=num_classes).astype(np.float32)
    total = float(counts.sum())
    weights = np.zeros(num_classes, dtype=np.float32)
    for class_id, count in enumerate(counts):
        if count > 0.0:
            weights[class_id] = total / (float(num_classes) * float(count))
    return torch.from_numpy(weights)


def class_counts_from_labels(labels: IntArray, num_classes: int) -> np.ndarray:
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError(f"class labels must be in [0, {num_classes - 1}]")
    return np.bincount(labels.astype(np.int64, copy=False), minlength=num_classes).astype(np.float32)


def class_weights_from_counts(counts: np.ndarray, mode: str, beta: float) -> torch.Tensor | None:
    if mode == "none":
        return None
    weights = np.zeros_like(counts, dtype=np.float32)
    present = counts > 0.0
    if not bool(present.any()):
        return torch.from_numpy(weights)
    if mode == "inverse_frequency":
        total = float(counts[present].sum())
        weights[present] = total / (float(len(counts)) * counts[present])
    elif mode == "effective_number":
        effective_num = 1.0 - np.power(beta, counts[present])
        weights[present] = (1.0 - beta) / np.maximum(effective_num, np.float32(1e-8))
        weights[present] = weights[present] / weights[present].sum() * float(len(counts))
    else:
        raise ValueError(f"Unsupported class weighting mode: {mode}")
    return torch.from_numpy(weights.astype(np.float32, copy=False))


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None, gamma: float, label_smoothing: float) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing)
        if self.gamma == 0.0:
            return cast(torch.Tensor, ce.mean())
        probs = torch.softmax(logits, dim=1)
        target_prob = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal = torch.pow(1.0 - target_prob, self.gamma) * ce
        return cast(torch.Tensor, focal.mean())


def classification_metrics(labels: list[int], preds: list[int], num_classes: int) -> dict[str, float]:
    total = len(labels)
    correct = sum(1 for label, pred in zip(labels, preds, strict=True) if label == pred)
    f1s: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    supports: list[int] = []
    for class_id in range(num_classes):
        tp = sum(1 for label, pred in zip(labels, preds, strict=True) if label == class_id and pred == class_id)
        fp = sum(1 for label, pred in zip(labels, preds, strict=True) if label != class_id and pred == class_id)
        fn = sum(1 for label, pred in zip(labels, preds, strict=True) if label == class_id and pred != class_id)
        support = sum(1 for label in labels if label == class_id)
        precision = safe_div(float(tp), float(tp + fp))
        recall = safe_div(float(tp), float(tp + fn))
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
    return {
        "accuracy": safe_div(float(correct), float(total)),
        "macro_f1": sum(f1s) / float(num_classes),
        "weighted_f1": safe_div(sum(f1 * support for f1, support in zip(f1s, supports, strict=True)), float(total)),
        "macro_precision": sum(precisions) / float(num_classes),
        "macro_recall": sum(recalls) / float(num_classes),
        "support": float(total),
    }


def zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


class MultiOutputLoss:
    def __init__(self, config: TrainConfig, train_dataset: MultiOutputCSIDataset, device: torch.device) -> None:
        presence_values = train_dataset.presence.astype(np.float32)
        positives = float(presence_values.sum())
        negatives = float(len(presence_values) - positives)
        pos_weight = torch.tensor([safe_div(negatives, positives)], dtype=torch.float32, device=device) if config.use_presence_pos_weight and positives > 0.0 else None
        valid_cell_labels = train_dataset.cell[(train_dataset.presence == 1) & (train_dataset.cell >= 0) & (train_dataset.cell < 25)]
        cell_counts = class_counts_from_labels(valid_cell_labels, 25)
        cell_weights = class_weights_from_counts(cell_counts, config.cell_class_weighting, config.cell_class_balance_beta)
        cell_weights = cell_weights.to(device) if cell_weights is not None else None
        pose_weights = class_weights_from_labels(train_dataset.pose, 7).to(device)
        self.presence_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if config.cell_loss_type == "focal":
            self.cell_loss: nn.Module = FocalCrossEntropyLoss(cell_weights, gamma=config.cell_focal_gamma, label_smoothing=config.cell_label_smoothing)
        else:
            self.cell_loss = nn.CrossEntropyLoss(weight=cell_weights, label_smoothing=config.cell_label_smoothing)
        self.pose_loss = nn.CrossEntropyLoss(weight=pose_weights)
        self.center_loss = nn.SmoothL1Loss()
        self.weights = {
            "presence": config.loss_weight_presence,
            "cell": config.loss_weight_cell,
            "pose": config.loss_weight_pose,
            "center": config.loss_weight_center,
        }

    def __call__(self, outputs: dict[str, torch.Tensor], target: Target) -> dict[str, torch.Tensor]:
        presence = target["presence"]
        cell_mask = target["cell_mask"]
        center_mask = target["center_mask"]
        losses = {
            "presence": self.presence_loss(outputs["presence_logit"], presence),
            "pose": self.pose_loss(outputs["pose_logits"], target["pose"]),
        }
        losses["cell"] = self.cell_loss(outputs["cell_logits"][cell_mask], target["human_cell"][cell_mask]) if bool(cell_mask.any()) else zero_loss(outputs["cell_logits"])
        losses["center"] = self.center_loss(outputs["center_norm"][center_mask], target["center_norm"][center_mask]) if bool(center_mask.any()) else zero_loss(outputs["center_norm"])
        total = zero_loss(outputs["presence_logit"])
        for name in self.weights:
            total = total + losses[name] * self.weights[name]
        losses["total"] = total
        return losses


def move_target(target: Target, device: torch.device) -> Target:
    return {key: value.to(device, non_blocking=True) for key, value in target.items()}


def run_epoch(model: nn.Module, loader: DataLoader[Batch], criterion: MultiOutputLoss, device: torch.device, optimizer: torch.optim.Optimizer | None, grad_clip: float, max_batches: int | None, disable_progress: bool, desc: str) -> dict[str, float]:
    training = optimizer is not None
    _ = model.train(training)
    totals = {"total": 0.0, "presence": 0.0, "cell": 0.0, "pose": 0.0, "center": 0.0}
    seen = 0
    rows: list[Row] = []
    iterator = tqdm(limited_iter(loader, max_batches), desc=desc, leave=False, disable=disable_progress)
    for x_batch, target_batch in iterator:
        x = x_batch.to(device, non_blocking=True)
        target = move_target(target_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = cast(dict[str, torch.Tensor], model(x))
            losses = criterion(outputs, target)
            if training:
                _ = losses["total"].backward()
                _ = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        batch_size = int(x.shape[0])
        seen += batch_size
        for key in totals:
            totals[key] += float(losses[key].item()) * batch_size
        rows.extend(batch_to_rows("", 0, outputs, target))
        _ = iterator.set_postfix(loss=totals["total"] / max(seen, 1))
    metrics = metrics_from_rows(rows)
    for key, value in totals.items():
        metrics[f"loss_{key}"] = value / max(seen, 1)
    metrics["loss"] = metrics["loss_total"]
    return metrics


def tensor_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in cast(Sequence[float], tensor.detach().cpu().tolist())]


def batch_to_rows(split: str, start_index: int, outputs: dict[str, torch.Tensor], target: Target) -> list[Row]:
    presence_probs = torch.sigmoid(outputs["presence_logit"]).detach().cpu()
    cell_probs = torch.softmax(outputs["cell_logits"], dim=1).detach().cpu()
    pose_probs = torch.softmax(outputs["pose_logits"], dim=1).detach().cpu()
    center_pred = outputs["center_norm"].detach().cpu()
    cpu_target = {key: value.detach().cpu() for key, value in target.items()}
    rows: list[Row] = []
    for idx in range(int(presence_probs.shape[0])):
        presence_prob = float(presence_probs[idx].item())
        cell_list = tensor_list(cell_probs[idx])
        pose_list = tensor_list(pose_probs[idx])
        cell_pred = int(np.argmax(cell_list))
        pose_pred = int(np.argmax(pose_list))
        center_label_norm = tensor_list(cpu_target["center_norm"][idx])
        center_label_m = tensor_list(cpu_target["center_m"][idx])
        center_pred_norm = tensor_list(center_pred[idx])
        center_pred_m = [value * CENTER_MAX_M for value in center_pred_norm]
        center_error = float(((center_pred_m[0] - center_label_m[0]) ** 2 + (center_pred_m[1] - center_label_m[1]) ** 2) ** 0.5) if int(cpu_target["center_mask"][idx].item()) else 0.0
        row: Row = {
            "split": split,
            "index": start_index + idx,
            "presence_label": int(cpu_target["presence"][idx].item()),
            "presence_prob": presence_prob,
            "presence_pred": int(presence_prob >= 0.5),
            "cell_label_raw": int(cpu_target["raw_cell"][idx].item()),
            "cell_label_human": int(cpu_target["human_cell"][idx].item()),
            "cell_pred": cell_pred,
            "cell_confidence": cell_list[cell_pred],
            "pose_label": int(cpu_target["pose"][idx].item()),
            "pose_pred": pose_pred,
            "pose_confidence": pose_list[pose_pred],
            "center_x_m_label": center_label_m[0],
            "center_y_m_label": center_label_m[1],
            "center_x_norm_label": center_label_norm[0],
            "center_y_norm_label": center_label_norm[1],
            "center_x_norm_pred": center_pred_norm[0],
            "center_y_norm_pred": center_pred_norm[1],
            "center_x_m_pred": center_pred_m[0],
            "center_y_m_pred": center_pred_m[1],
            "center_error_m": center_error,
        }
        row.update({f"cell_prob_{class_id}": cell_list[class_id] for class_id in range(25)})
        row.update({f"visual_cell_prob_{class_id}": presence_prob * cell_list[class_id] for class_id in range(25)})
        row.update({f"pose_prob_{class_id}": pose_list[class_id] for class_id in range(7)})
        rows.append(row)
    return rows


def metrics_from_rows(rows: list[Row]) -> dict[str, float]:
    presence_labels = [cast(int, row["presence_label"]) for row in rows]
    presence_preds = [cast(int, row["presence_pred"]) for row in rows]
    pose_labels = [cast(int, row["pose_label"]) for row in rows]
    pose_preds = [cast(int, row["pose_pred"]) for row in rows]
    cell_rows = [row for row in rows if cast(int, row["cell_label_human"]) >= 0]
    center_rows = [row for row in rows if cast(int, row["presence_label"]) == 1]
    metrics: dict[str, float] = {}
    for prefix, values in (("presence", classification_metrics(presence_labels, presence_preds, 2)), ("pose", classification_metrics(pose_labels, pose_preds, 7))):
        metrics.update({f"{prefix}_{key}": value for key, value in values.items()})
    if cell_rows:
        cell_labels = [cast(int, row["cell_label_human"]) for row in cell_rows]
        cell_preds = [cast(int, row["cell_pred"]) for row in cell_rows]
        metrics.update({f"cell_masked_{key}": value for key, value in classification_metrics(cell_labels, cell_preds, 25).items()})
        cell_l1_distances: list[int] = []
        cell_linf_distances: list[int] = []
        cell_soft_scores: list[float] = []
        for label, pred in zip(cell_labels, cell_preds, strict=True):
            label_row, label_col = divmod(label, 5)
            pred_row, pred_col = divmod(pred, 5)
            row_distance = abs(label_row - pred_row)
            col_distance = abs(label_col - pred_col)
            l1_distance = row_distance + col_distance
            linf_distance = max(row_distance, col_distance)
            cell_l1_distances.append(l1_distance)
            cell_linf_distances.append(linf_distance)
            if linf_distance == 0:
                cell_soft_scores.append(1.0)
            elif l1_distance == 1:
                cell_soft_scores.append(0.4)
            elif row_distance == 1 and col_distance == 1:
                cell_soft_scores.append(0.3)
            else:
                cell_soft_scores.append(0.0)
        metrics["cell_relaxed_soft_score"] = float(sum(cell_soft_scores) / len(cell_soft_scores))
        metrics["cell_within_1cell"] = float(sum(distance <= 1 for distance in cell_linf_distances) / len(cell_linf_distances))
        metrics["cell_within_2cell"] = float(sum(distance <= 2 for distance in cell_linf_distances) / len(cell_linf_distances))
        metrics["cell_grid_l1_mean"] = float(sum(cell_l1_distances) / len(cell_l1_distances))
        metrics["cell_grid_linf_mean"] = float(sum(cell_linf_distances) / len(cell_linf_distances))
    else:
        metrics.update({"cell_masked_accuracy": 0.0, "cell_masked_macro_f1": 0.0, "cell_masked_weighted_f1": 0.0, "cell_masked_macro_precision": 0.0, "cell_masked_macro_recall": 0.0, "cell_masked_support": 0.0, "cell_relaxed_soft_score": 0.0, "cell_within_1cell": 0.0, "cell_within_2cell": 0.0, "cell_grid_l1_mean": 0.0, "cell_grid_linf_mean": 0.0})
    if center_rows:
        norm_abs = [
            abs(cast(float, row["center_x_norm_pred"]) - cast(float, row["center_x_norm_label"]))
            + abs(cast(float, row["center_y_norm_pred"]) - cast(float, row["center_y_norm_label"]))
            for row in center_rows
        ]
        norm_sq = [
            (cast(float, row["center_x_norm_pred"]) - cast(float, row["center_x_norm_label"])) ** 2
            + (cast(float, row["center_y_norm_pred"]) - cast(float, row["center_y_norm_label"])) ** 2
            for row in center_rows
        ]
        meter_errors = [cast(float, row["center_error_m"]) for row in center_rows]
        metrics["center_norm_mae"] = float(sum(norm_abs) / len(norm_abs))
        metrics["center_m_rmse"] = float((sum(error * error for error in meter_errors) / len(meter_errors)) ** 0.5)
        metrics["center_norm_rmse"] = float((sum(norm_sq) / len(norm_sq)) ** 0.5)
        metrics["center_score"] = max(0.0, 1.0 - metrics["center_norm_rmse"])
        metrics["center_mean_error_m"] = sum(meter_errors) / len(meter_errors)
        metrics["center_support"] = float(len(center_rows))
    else:
        metrics.update({"center_norm_mae": 0.0, "center_m_rmse": 0.0, "center_norm_rmse": 0.0, "center_score": 0.0, "center_mean_error_m": 0.0, "center_support": 0.0})
    return metrics


def predict(model: nn.Module, loader: DataLoader[Batch], criterion: MultiOutputLoss, device: torch.device, max_batches: int | None, disable_progress: bool, split: str) -> tuple[list[Row], dict[str, float]]:
    _ = model.eval()
    rows: list[Row] = []
    totals = {"total": 0.0, "presence": 0.0, "cell": 0.0, "pose": 0.0, "center": 0.0}
    seen = 0
    sample_index = 0
    with torch.no_grad():
        iterator = tqdm(limited_iter(loader, max_batches), desc=f"predict {split}", leave=False, disable=disable_progress)
        for x_batch, target_batch in iterator:
            x = x_batch.to(device, non_blocking=True)
            target = move_target(target_batch, device)
            outputs = cast(dict[str, torch.Tensor], model(x))
            losses = criterion(outputs, target)
            batch_size = int(x.shape[0])
            for key in totals:
                totals[key] += float(losses[key].item()) * batch_size
            rows.extend(batch_to_rows(split, sample_index, outputs, target))
            sample_index += batch_size
            seen += batch_size
    metrics = metrics_from_rows(rows)
    for key, value in totals.items():
        metrics[f"loss_{key}"] = value / max(seen, 1)
    metrics["loss"] = metrics["loss_total"]
    return rows, metrics


def composite_score(metrics: dict[str, float]) -> float:
    return (metrics["presence_macro_f1"] + metrics["cell_masked_macro_f1"] + metrics["pose_macro_f1"] + metrics["center_score"]) / 4.0


def is_better(val_metrics: dict[str, float], best_score: float, best_loss: float) -> bool:
    score = composite_score(val_metrics)
    loss = val_metrics["loss"]
    return score > best_score or (score == best_score and loss < best_loss)


def write_csv(path: Path, rows: list[Row], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: ReduceLROnPlateau, epoch: int, best_score: float, config: TrainConfig) -> None:
    _ = torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "epoch": epoch, "best_metric": best_score, "best_metric_name": "val_composite_score", "config": asdict(config)}, path)


def select_best_attention_gru(run_prefixes: tuple[str, ...] = ("A", "S2_A")) -> tuple[dict[str, object], dict[str, object]]:
    fallback: dict[str, object] = {"profile": "raw_shift", "hidden_size": 96, "attention_dim": 32, "dropout": 0.2, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "noise_sigma": 0.0, "num_layers": 1, "bidirectional": False, "scheduler_factor": 0.5, "scheduler_patience": None, "scheduler_min_lr": 0.0}
    candidates: list[tuple[float, float, Path, dict[str, object], dict[str, object]]] = []
    for metrics_path in ATTENTION_RUNS_DIR.glob("*/metrics.json"):
        config_path = metrics_path.parent / "config.json"
        best_model_path = metrics_path.parent / "best_model.pt"
        if not config_path.is_file() or not best_model_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        experiment_name = str(config.get("experiment_name", metrics.get("experiment_name", metrics_path.parent.name)))
        folder_name = metrics_path.parent.name
        if "smoke" in experiment_name.lower() or "smoke" in folder_name.lower():
            continue
        if not any(experiment_name.startswith(prefix) or folder_name.startswith(prefix) for prefix in run_prefixes):
            continue
        if metrics.get("phase_name") != "Attention-GRU" and config.get("phase_name") != "Attention-GRU":
            continue
        if config.get("max_train_batches") is not None or config.get("max_eval_batches") is not None:
            continue
        if int(metrics.get("epochs_ran", 0)) <= 1:
            continue
        if "best_val_macro_f1" not in metrics:
            continue
        candidates.append((float(metrics["best_val_macro_f1"]), float(metrics.get("best_val_loss", float("inf"))), metrics_path.parent, config, metrics))
    if not candidates:
        return fallback, {"selected": False, "reason": f"no completed planned Attention-GRU metrics found for prefixes {run_prefixes}; using B2 defaults", "source_run_dir": None, "selected_config": fallback, "run_prefixes": list(run_prefixes)}
    candidates.sort(key=lambda item: (-item[0], item[1], str(item[2])))
    macro_f1, val_loss, run_dir, config, metrics = candidates[0]
    selected = {key: config.get(key, fallback[key]) for key in fallback}
    selected["profile"] = config.get("profile", fallback["profile"])
    return selected, {"selected": True, "reason": "best completed planned Attention-GRU by best_val_macro_f1 with best_val_loss tie-break", "source_run_dir": str(run_dir), "source_best_val_macro_f1": macro_f1, "source_best_val_loss": val_loss, "source_metrics": {"experiment_name": metrics.get("experiment_name"), "best_epoch": metrics.get("best_epoch")}, "selected_config": selected, "run_prefixes": list(run_prefixes)}


def apply_selection(config: TrainConfig) -> dict[str, object]:
    selected, report = select_best_attention_gru(config.selector_run_prefixes)
    for key, value in selected.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return report


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def prediction_fields() -> list[str]:
    base = ["split", "index", "presence_label", "presence_prob", "presence_pred", "cell_label_raw", "cell_label_human", "cell_pred", "cell_confidence", "pose_label", "pose_pred", "pose_confidence", "center_x_m_label", "center_y_m_label", "center_x_norm_label", "center_y_norm_label", "center_x_norm_pred", "center_y_norm_pred", "center_x_m_pred", "center_y_m_pred", "center_error_m"]
    return base + [f"cell_prob_{idx}" for idx in range(25)] + [f"visual_cell_prob_{idx}" for idx in range(25)] + [f"pose_prob_{idx}" for idx in range(7)]


def train(config: TrainConfig, use_selector: bool = True) -> Row:
    _ = validate_safe_name(config.experiment_name, "experiment_name")
    _ = validate_safe_name(config.profile, "profile")
    if config.profile not in PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(sorted(PROFILES))}")
    if use_selector:
        selection = apply_selection(config)
    else:
        selection = {"selected": False, "reason": "selector disabled by caller", "source_run_dir": None, "selected_config": {}}
    validate_config(config)
    set_seed(config.seed)
    run_dir = resolve_run_dir(config.experiment_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    _ = (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")
    _ = (run_dir / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    loaders = make_dataloaders(config)
    train_dataset = cast(MultiOutputCSIDataset, loaders["train"].dataset)
    device = resolve_device(config.device)
    model = MultiOutputAttentionGRU(
        hidden_size=config.hidden_size,
        attention_dim=config.attention_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        cell_head_type=config.cell_head_type,
        cell_head_hidden_size=config.cell_head_hidden_size,
        cell_head_dropout=config.cell_head_dropout,
    ).to(device)
    criterion = MultiOutputLoss(config, train_dataset, device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(config.optimizer_beta1, config.optimizer_beta2), eps=config.optimizer_eps)
    scheduler_patience = config.scheduler_patience if config.scheduler_patience is not None else max(2, config.patience // 3)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=config.scheduler_factor, patience=scheduler_patience, min_lr=config.scheduler_min_lr)
    best_score = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[Row] = []
    epoch_bar = tqdm(range(1, config.epochs + 1), desc=config.phase_name, disable=config.disable_progress)
    for epoch in epoch_bar:
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer, config.grad_clip, config.max_train_batches, config.disable_progress, f"train {epoch}")
        val_metrics = run_epoch(model, loaders["val"], criterion, device, None, config.grad_clip, config.max_eval_batches, config.disable_progress, f"val {epoch}")
        val_score = composite_score(val_metrics)
        _ = scheduler.step(val_score)
        current_best = is_better(val_metrics, best_score, best_loss)
        if current_best:
            best_score = val_score
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(run_dir / "best_model.pt", model, optimizer, scheduler, epoch, best_score, config)
        else:
            stale_epochs += 1
        save_checkpoint(run_dir / "last_model.pt", model, optimizer, scheduler, epoch, best_score, config)
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"], "val_composite_score": val_score, "val_presence_f1": val_metrics["presence_macro_f1"], "val_cell_masked_macro_f1": val_metrics["cell_masked_macro_f1"], "val_pose_macro_f1": val_metrics["pose_macro_f1"], "val_center_score": val_metrics["center_score"], "lr": optimizer.param_groups[0]["lr"], "is_best": current_best})
        _ = epoch_bar.set_postfix(score=val_score, loss=val_metrics["loss"])
        if stale_epochs >= config.patience:
            break
    write_csv(run_dir / "history.csv", history, ["epoch", "train_loss", "val_loss", "val_composite_score", "val_presence_f1", "val_cell_masked_macro_f1", "val_pose_macro_f1", "val_center_score", "lr", "is_best"])
    best_state = cast(dict[str, object], torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    _ = model.load_state_dict(cast(dict[str, torch.Tensor], best_state["model_state"]))
    val_rows, val_metrics = predict(model, loaders["val"], criterion, device, config.max_eval_batches, config.disable_progress, "val")
    test_rows, test_metrics = predict(model, loaders["test"], criterion, device, config.max_eval_batches, config.disable_progress, "test")
    write_csv(run_dir / "predictions.csv", val_rows + test_rows, prediction_fields())
    metrics: Row = {"phase_name": config.phase_name, "experiment_name": config.experiment_name, "profile": config.profile, "hidden_size": config.hidden_size, "attention_dim": config.attention_dim, "dropout": config.dropout, "batch_size": config.batch_size, "learning_rate": config.learning_rate, "weight_decay": config.weight_decay, "noise_sigma": config.noise_sigma, "num_layers": config.num_layers, "bidirectional": config.bidirectional, "cell_head_type": config.cell_head_type, "cell_head_hidden_size": config.cell_head_hidden_size, "cell_head_dropout": config.cell_head_dropout, "cell_loss_type": config.cell_loss_type, "cell_label_smoothing": config.cell_label_smoothing, "cell_focal_gamma": config.cell_focal_gamma, "cell_class_weighting": config.cell_class_weighting, "cell_class_balance_beta": config.cell_class_balance_beta, "best_epoch": best_epoch, "best_metric_name": "val_composite_score", "best_val_composite_score": best_score, "epochs_ran": len(history), "run_dir": str(run_dir), "selection": selection}
    metrics.update(prefixed("val", val_metrics))
    metrics.update(prefixed("test", test_metrics))
    _ = (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phase-specific multi-output Attention-GRU for CSI Part 3 metrics and 2.5D probability cloud exports.")
    _ = parser.add_argument("--experiment-name", default="multi_output_best_model")
    _ = parser.add_argument("--profile", default=None)
    _ = parser.add_argument("--max-epochs", type=int, default=None)
    _ = parser.add_argument("--max-train-batches", type=int, default=None)
    _ = parser.add_argument("--max-eval-batches", type=int, default=None)
    _ = parser.add_argument("--device", default="auto")
    _ = parser.add_argument("--disable-progress", action="store_true")
    _ = parser.add_argument("--no-selector", action="store_true")
    _ = parser.add_argument("--run-full", action="store_true")
    _ = parser.add_argument("--learning-rate", type=float, default=None)
    _ = parser.add_argument("--dropout", type=float, default=None)
    _ = parser.add_argument("--batch-size", type=int, default=None)
    _ = parser.add_argument("--hidden-size", type=int, default=None)
    _ = parser.add_argument("--attention-dim", type=int, default=None)
    _ = parser.add_argument("--num-layers", type=int, default=None)
    _ = parser.add_argument("--bidirectional", action="store_true")
    _ = parser.add_argument("--cell-head-type", choices=["linear", "mlp"], default=None)
    _ = parser.add_argument("--cell-loss-type", choices=["cross_entropy", "focal"], default=None)
    _ = parser.add_argument("--cell-focal-gamma", type=float, default=None)
    _ = parser.add_argument("--cell-label-smoothing", type=float, default=None)
    _ = parser.add_argument("--cell-class-weighting", choices=["none", "inverse_frequency", "effective_number"], default=None)
    _ = parser.add_argument("--loss-weight-cell", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(experiment_name=cast(str, args.experiment_name))
    if cast(str | None, args.profile) is not None:
        config.profile = cast(str, args.profile)
    if cast(int | None, args.max_epochs) is not None:
        config.epochs = cast(int, args.max_epochs)
    config.max_train_batches = cast(int | None, args.max_train_batches)
    config.max_eval_batches = cast(int | None, args.max_eval_batches)
    config.device = cast(str, args.device)
    config.disable_progress = cast(bool, args.disable_progress)
    learning_rate = cast(float | None, args.learning_rate)
    dropout = cast(float | None, args.dropout)
    batch_size = cast(int | None, args.batch_size)
    hidden_size = cast(int | None, args.hidden_size)
    attention_dim = cast(int | None, args.attention_dim)
    num_layers = cast(int | None, args.num_layers)
    cell_head_type = cast(str | None, args.cell_head_type)
    cell_loss_type = cast(str | None, args.cell_loss_type)
    cell_focal_gamma = cast(float | None, args.cell_focal_gamma)
    cell_label_smoothing = cast(float | None, args.cell_label_smoothing)
    cell_class_weighting = cast(str | None, args.cell_class_weighting)
    loss_weight_cell = cast(float | None, args.loss_weight_cell)
    if learning_rate is not None:
        config.learning_rate = learning_rate
    if dropout is not None:
        config.dropout = dropout
    if batch_size is not None:
        config.batch_size = batch_size
    if hidden_size is not None:
        config.hidden_size = hidden_size
    if attention_dim is not None:
        config.attention_dim = attention_dim
    if num_layers is not None:
        config.num_layers = num_layers
    if cast(bool, args.bidirectional):
        config.bidirectional = True
    if cell_head_type is not None:
        config.cell_head_type = cell_head_type
    if cell_loss_type is not None:
        config.cell_loss_type = cell_loss_type
    if cell_focal_gamma is not None:
        config.cell_focal_gamma = cell_focal_gamma
    if cell_label_smoothing is not None:
        config.cell_label_smoothing = cell_label_smoothing
    if cell_class_weighting is not None:
        config.cell_class_weighting = cell_class_weighting
    if loss_weight_cell is not None:
        config.loss_weight_cell = loss_weight_cell
    if not cast(bool, args.run_full) and config.max_train_batches is None:
        raise ValueError("Refusing unbounded CLI training without --run-full; pass --max-train-batches for smoke checks or --run-full for intentional full training.")
    metrics = train(config, use_selector=not cast(bool, args.no_selector))
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
