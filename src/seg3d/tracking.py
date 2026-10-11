import csv
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIELDS = (
    "epoch", "train_loss", "dice_wt", "dice_tc", "dice_et", "mean_dice",
    "learning_rate", "epoch_seconds", "peak_vram_gib",
)


def setup_logging(path):
    logger = logging.getLogger("seg3d")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    training_handler = logging.FileHandler(path)
    training_handler.setLevel(logging.INFO)
    training_handler.setFormatter(formatter)
    logger.addHandler(training_handler)

    # The launcher redirects stdout/stderr to console.log. Routine INFO stays
    # only in training.log; operational warnings and errors are echoed there.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def append_metrics(path, row):
    exists = path.exists()
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


def read_metrics(path):
    if not path.exists():
        return []
    with open(path, newline="") as file:
        return [
            {key: int(value) if key == "epoch" else float(value)
             for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def update_curves(csv_path: Path, output_path: Path):
    rows = read_metrics(csv_path)
    if not rows:
        return
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], color="tab:blue")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Dice + BCE")
    axes[0].grid(alpha=0.25)
    for key, label in (
        ("dice_wt", "WT"), ("dice_tc", "TC"), ("dice_et", "ET"),
        ("mean_dice", "Mean"),
    ):
        axes[1].plot(epochs, [row[key] for row in rows], label=label)
    axes[1].set(title="Validation Dice", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.suptitle(output_path.parents[1].name)
    figure.tight_layout()
    temporary = output_path.with_suffix(".tmp")
    figure.savefig(temporary, format="png", dpi=150)
    plt.close(figure)
    os.replace(temporary, output_path)
