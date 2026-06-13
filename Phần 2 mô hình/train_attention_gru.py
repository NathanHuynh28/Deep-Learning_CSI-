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
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from attention_gru import AttentionGRUPoseClassifier

Batch = tuple[torch.Tensor, torch.Tensor]
Row = dict[str, object]
FloatMetrics = dict[str, float]

ROOT_DIR = Path(__file__).resolve().parents[1]
PHASE2_DIR = Path(__file__).resolve().parent
PART1_DIR = ROOT_DIR / "Phần 1 tiền xử lí"
DATASET_PATH = PART1_DIR / "Dataset" / "CSI_dataset.py"
WINDOWS_DIR = PART1_DIR / "processed" / "windows"
RUNS_DIR = PHASE2_DIR / "runs" / "attention_gru"
NUM_CLASSES = 7
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ATTENTION_GRU_PROFILES = {"raw", "raw_shift", "phase_hampel", "phase_hampel_shift"}


@dataclass
class TrainConfig:
    phase_name: str = "Attention-GRU"
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
    experiment_name: str = "B2_raw_shift_attgru_h96_a32"
    device: str = "auto"
    disable_progress: bool = False
    max_train_batches: int | None = None
    max_eval_batches: int | None = None


def experiment_configs() -> dict[str, TrainConfig]:
    return {
        "B1_raw_attgru_h96_a32": TrainConfig(profile="raw", noise_sigma=0.0, experiment_name="B1_raw_attgru_h96_a32"),
        "B2_raw_shift_attgru_h96_a32": TrainConfig(profile="raw_shift", noise_sigma=0.0, experiment_name="B2_raw_shift_attgru_h96_a32"),
        "B3_raw_shift_noise001_attgru_h96_a32": TrainConfig(profile="raw_shift", noise_sigma=0.01, experiment_name="B3_raw_shift_noise001_attgru_h96_a32"),
        "B4_phase_hampel_attgru_h96_a32": TrainConfig(profile="phase_hampel", noise_sigma=0.0, experiment_name="B4_phase_hampel_attgru_h96_a32"),
        "B5_phase_hampel_shift_attgru_h96_a32": TrainConfig(profile="phase_hampel_shift", noise_sigma=0.0, experiment_name="B5_phase_hampel_shift_attgru_h96_a32"),
        "B6_phase_hampel_shift_noise001_attgru_h96_a32": TrainConfig(profile="phase_hampel_shift", noise_sigma=0.01, experiment_name="B6_phase_hampel_shift_noise001_attgru_h96_a32"),
    }


class DatasetModule(Protocol):
    def make_dataloaders(
        self,
        windows_dir: Path,
        profile: str,
        target: str,
        batch_size: int,
        num_workers: int,
        normalize_runtime: bool,
        noise_sigma: float,
    ) -> dict[str, DataLoader[Batch]]: ...

    def class_weights(self, profile_dir: Path, split: str = "train", target: str = "pose") -> torch.Tensor: ...


def load_dataset_module() -> DatasetModule:
    spec = importlib.util.spec_from_file_location("csi_dataset", DATASET_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dataset module from {DATASET_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(DatasetModule, cast(object, module))


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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
    if config.phase_name != "Attention-GRU":
        raise ValueError("Attention-GRU training requires phase_name='Attention-GRU'")
    _ = validate_safe_name(config.experiment_name, "experiment_name")
    _ = validate_safe_name(config.profile, "profile")
    if config.profile not in ATTENTION_GRU_PROFILES:
        allowed_profiles = ", ".join(sorted(ATTENTION_GRU_PROFILES))
        raise ValueError(f"Attention-GRU training requires profile to be one of: {allowed_profiles}")
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.patience < 1:
        raise ValueError("patience must be >= 1")
    if config.hidden_size < 1:
        raise ValueError("hidden_size must be >= 1")
    if config.attention_dim < 1:
        raise ValueError("attention_dim must be >= 1")
    if config.num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must satisfy 0 <= dropout < 1")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be >= 0")
    if config.grad_clip <= 0.0:
        raise ValueError("grad_clip must be > 0")
    if config.noise_sigma < 0.0:
        raise ValueError("noise_sigma must be >= 0")
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
    if config.max_train_batches is not None and config.max_train_batches < 1:
        raise ValueError("max_train_batches must be None or >= 1")
    if config.max_eval_batches is not None and config.max_eval_batches < 1:
        raise ValueError("max_eval_batches must be None or >= 1")


def pose_class_weights(raw_weights: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    weights = raw_weights.detach().to(dtype=torch.float32).flatten()
    if int(weights.numel()) > num_classes:
        raise ValueError(f"pose class_weights has {int(weights.numel())} entries, expected at most {num_classes}")
    if int(weights.numel()) < num_classes:
        padded = torch.zeros(num_classes, dtype=torch.float32)
        padded[: int(weights.numel())] = weights
        return padded
    return weights


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    _ = torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_iter(loader: DataLoader[Batch], max_batches: int | None) -> Iterator[Batch]:
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield cast(Batch, batch)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def classification_metrics(labels: list[int], predictions: list[int], num_classes: int = NUM_CLASSES) -> FloatMetrics:
    total = len(labels)
    correct = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == pred)
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    supports: list[int] = []
    for class_id in range(num_classes):
        true_positive = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == class_id and pred == class_id)
        false_positive = sum(1 for label, pred in zip(labels, predictions, strict=True) if label != class_id and pred == class_id)
        false_negative = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == class_id and pred != class_id)
        support = sum(1 for label in labels if label == class_id)
        precision = safe_div(float(true_positive), float(true_positive + false_positive))
        recall = safe_div(float(true_positive), float(true_positive + false_negative))
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
    return {
        "accuracy": safe_div(float(correct), float(total)),
        "macro_f1": sum(f1_scores) / num_classes,
        "weighted_f1": safe_div(sum(score * support for score, support in zip(f1_scores, supports, strict=True)), float(total)),
        "macro_precision": sum(precisions) / num_classes,
        "macro_recall": sum(recalls) / num_classes,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Batch],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    max_batches: int | None,
    disable_progress: bool,
    desc: str,
) -> FloatMetrics:
    training = optimizer is not None
    _ = model.train(training)
    total_loss = 0.0
    total_seen = 0
    labels: list[int] = []
    predictions: list[int] = []
    iterator = tqdm(limited_iter(loader, max_batches), desc=desc, leave=False, disable=disable_progress)
    for x_batch, y_batch in iterator:
        x = x_batch.to(device, non_blocking=True)
        y = y_batch.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = cast(torch.Tensor, model(x))
            loss = cast(torch.Tensor, criterion(logits, y))
            if training:
                _ = loss.backward()
                _ = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        batch_size = int(y.numel())
        batch_preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * batch_size
        total_seen += batch_size
        label_values = cast(Sequence[int], y.detach().cpu().tolist())
        pred_values = cast(Sequence[int], batch_preds.detach().cpu().tolist())
        labels.extend(int(value) for value in label_values)
        predictions.extend(int(value) for value in pred_values)
        if total_seen:
            interim = classification_metrics(labels, predictions)
            _ = iterator.set_postfix(loss=total_loss / total_seen, f1=interim["macro_f1"])
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = total_loss / max(total_seen, 1)
    return metrics


def predict(
    model: nn.Module,
    loader: DataLoader[Batch],
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
    disable_progress: bool,
    split: str,
) -> tuple[list[Row], FloatMetrics]:
    _ = model.eval()
    rows: list[Row] = []
    labels: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    total_seen = 0
    sample_index = 0
    iterator = tqdm(limited_iter(loader, max_batches), desc=f"predict {split}", leave=False, disable=disable_progress)
    with torch.no_grad():
        for x_batch, y_batch in iterator:
            y = y_batch.to(device, non_blocking=True)
            logits = cast(torch.Tensor, model(x_batch.to(device, non_blocking=True)))
            loss = cast(torch.Tensor, criterion(logits, y))
            probs = torch.softmax(logits, dim=1).detach().cpu()
            preds = probs.argmax(dim=1)
            cpu_labels = y.detach().cpu()
            batch_size = int(cpu_labels.numel())
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size
            label_values = [int(value) for value in cast(Sequence[int], cpu_labels.tolist())]
            pred_values = [int(value) for value in cast(Sequence[int], preds.tolist())]
            labels.extend(label_values)
            predictions.extend(pred_values)
            for label, pred, prob_tensor in zip(label_values, pred_values, probs, strict=True):
                probs_list = [float(value) for value in cast(Sequence[float], prob_tensor.tolist())]
                row: Row = {
                    "split": split,
                    "index": sample_index,
                    "label": label,
                    "prediction": pred,
                    "confidence": probs_list[pred],
                }
                row.update({f"pose_prob_{class_id}": probs_list[class_id] for class_id in range(NUM_CLASSES)})
                rows.append(row)
                sample_index += 1
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = total_loss / max(total_seen, 1)
    return rows, metrics


def confusion_matrix(rows: list[Row], num_classes: int = NUM_CLASSES) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for row in rows:
        label = cast(int, row["label"])
        prediction = cast(int, row["prediction"])
        matrix[label][prediction] += 1
    return matrix


def write_csv(path: Path, rows: list[Row], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_metric: float,
    config: TrainConfig,
) -> None:
    _ = torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "best_metric_name": "val_macro_f1",
            "config": asdict(config),
        },
        path,
    )


def is_better_candidate(val_metrics: FloatMetrics, best_macro_f1: float, best_val_loss: float) -> bool:
    macro_f1 = val_metrics["macro_f1"]
    val_loss = val_metrics["loss"]
    return macro_f1 > best_macro_f1 or (macro_f1 == best_macro_f1 and val_loss < best_val_loss)


def prefixed_metrics(prefix: str, metrics: FloatMetrics) -> Row:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def train(config: TrainConfig) -> Row:
    validate_config(config)
    set_seed(config.seed)
    run_dir = resolve_run_dir(config.experiment_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    _ = (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")

    dataset_module = load_dataset_module()
    loaders = dataset_module.make_dataloaders(
        windows_dir=WINDOWS_DIR,
        profile=config.profile,
        target="pose",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        normalize_runtime=config.normalize_runtime,
        noise_sigma=config.noise_sigma,
    )
    device = resolve_device(config.device)
    model = AttentionGRUPoseClassifier(
        hidden_size=config.hidden_size,
        attention_dim=config.attention_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        num_classes=NUM_CLASSES,
    ).to(device)
    raw_weights = dataset_module.class_weights(WINDOWS_DIR / config.profile, split="train", target="pose")
    weights = pose_class_weights(raw_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(config.optimizer_beta1, config.optimizer_beta2), eps=config.optimizer_eps)
    scheduler_patience = config.scheduler_patience if config.scheduler_patience is not None else max(2, config.patience // 3)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=config.scheduler_factor, patience=scheduler_patience, min_lr=config.scheduler_min_lr)

    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[Row] = []
    epoch_bar = tqdm(range(1, config.epochs + 1), desc=config.phase_name, disable=config.disable_progress)
    for epoch in epoch_bar:
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer, config.grad_clip, config.max_train_batches, config.disable_progress, f"train {epoch}")
        val_metrics = run_epoch(model, loaders["val"], criterion, device, None, config.grad_clip, config.max_eval_batches, config.disable_progress, f"val {epoch}")
        _ = scheduler.step(val_metrics["macro_f1"])
        is_best = is_better_candidate(val_metrics, best_macro_f1, best_val_loss)
        if is_best:
            best_macro_f1 = val_metrics["macro_f1"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(run_dir / "best_model.pt", model, optimizer, scheduler, epoch, best_macro_f1, config)
        else:
            stale_epochs += 1
        save_checkpoint(run_dir / "last_model.pt", model, optimizer, scheduler, epoch, best_macro_f1, config)
        row: Row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "lr": optimizer.param_groups[0]["lr"],
            "is_best": is_best,
        }
        history.append(row)
        _ = epoch_bar.set_postfix(val_f1=val_metrics["macro_f1"], val_loss=val_metrics["loss"])
        if stale_epochs >= config.patience:
            break

    history_fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "val_macro_precision",
        "val_macro_recall",
        "lr",
        "is_best",
    ]
    write_csv(run_dir / "history.csv", history, history_fields)

    best_state = cast(dict[str, object], torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    _ = model.load_state_dict(cast(dict[str, torch.Tensor], best_state["model_state"]))
    val_rows, final_val_metrics = predict(model, loaders["val"], criterion, device, config.max_eval_batches, config.disable_progress, "val")
    test_rows, final_test_metrics = predict(model, loaders["test"], criterion, device, config.max_eval_batches, config.disable_progress, "test")
    prediction_rows = val_rows + test_rows
    prediction_fields = ["split", "index", "label", "prediction", "confidence"] + [f"pose_prob_{idx}" for idx in range(NUM_CLASSES)]
    write_csv(run_dir / "predictions.csv", prediction_rows, prediction_fields)

    matrix = confusion_matrix(test_rows)
    with (run_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + [f"pred_{idx}" for idx in range(NUM_CLASSES)])
        for label, values in enumerate(matrix):
            writer.writerow([label] + values)

    metrics: Row = {
        "phase_name": config.phase_name,
        "profile": config.profile,
        "experiment_name": config.experiment_name,
        "hidden_size": config.hidden_size,
        "attention_dim": config.attention_dim,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "noise_sigma": config.noise_sigma,
        "num_layers": config.num_layers,
        "bidirectional": config.bidirectional,
        "best_epoch": best_epoch,
        "best_metric_name": "val_macro_f1",
        "epochs_ran": len(history),
        "run_dir": str(run_dir),
    }
    metrics.update(prefixed_metrics("best_val", final_val_metrics))
    metrics.update(prefixed_metrics("test", final_test_metrics))
    _ = (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Attention-GRU for CSI pose classification.")
    _ = parser.add_argument("--experiment", choices=sorted(experiment_configs()), default="B2_raw_shift_attgru_h96_a32")
    _ = parser.add_argument("--experiment-name", default=None)
    _ = parser.add_argument("--max-epochs", type=int, default=None)
    _ = parser.add_argument("--max-train-batches", type=int, default=None)
    _ = parser.add_argument("--max-eval-batches", type=int, default=None)
    _ = parser.add_argument("--device", default="auto")
    _ = parser.add_argument("--disable-progress", action="store_true")
    _ = parser.add_argument("--learning-rate", type=float, default=None)
    _ = parser.add_argument("--dropout", type=float, default=None)
    _ = parser.add_argument("--batch-size", type=int, default=None)
    _ = parser.add_argument("--hidden-size", type=int, default=None)
    _ = parser.add_argument("--attention-dim", type=int, default=None)
    _ = parser.add_argument("--num-layers", type=int, default=None)
    _ = parser.add_argument("--bidirectional", action="store_true")
    _ = parser.add_argument("--weight-decay", type=float, default=None)
    _ = parser.add_argument("--scheduler-factor", type=float, default=None)
    _ = parser.add_argument("--scheduler-patience", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = cast(str, args.experiment)
    config = experiment_configs()[experiment]
    experiment_name = cast(str | None, args.experiment_name)
    max_epochs = cast(int | None, args.max_epochs)
    if experiment_name is not None:
        config.experiment_name = experiment_name
    if max_epochs is not None:
        config.epochs = max_epochs
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
    weight_decay = cast(float | None, args.weight_decay)
    scheduler_factor = cast(float | None, args.scheduler_factor)
    scheduler_patience = cast(int | None, args.scheduler_patience)
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
    if weight_decay is not None:
        config.weight_decay = weight_decay
    if scheduler_factor is not None:
        config.scheduler_factor = scheduler_factor
    if scheduler_patience is not None:
        config.scheduler_patience = scheduler_patience
    metrics = train(config)
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
