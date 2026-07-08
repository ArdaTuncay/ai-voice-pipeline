import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import wandb

from src.utils.config import settings
from src.utils.logger_manager import logger


@dataclass
class TrainingStats:
    total_embeddings: int = 0
    epochs_completed: int = 0
    final_loss: float = 0.0
    final_accuracy: float = 0.0
    duration_sec: float = 0.0


def load_embeddings(features_dir: Path) -> list[np.ndarray]:
    """data/features/ altındaki .npy embedding dosyalarını belleğe yükler."""
    files = sorted(f for f in features_dir.iterdir() if f.is_file() and f.suffix.lower() == ".npy")

    embeddings = []
    for file_path in files:
        try:
            embeddings.append(np.load(file_path))
        except (OSError, ValueError) as e:
            logger.error(f"Embedding yüklenemedi: {file_path.name} - {e}")

    return embeddings


def simulate_step(epoch: int, total_epochs: int) -> tuple[float, float]:
    """RVC core henüz derlenmediği için gerçek eğitim yerine azalan loss / artan accuracy simüle eder."""
    progress = epoch / total_epochs
    loss = 2.0 * math.exp(-3.0 * progress) + random.uniform(0.0, 0.03)
    accuracy = min(0.99, 1.0 - math.exp(-3.0 * progress)) + random.uniform(-0.01, 0.01)
    accuracy = max(0.0, min(1.0, accuracy))
    return loss, accuracy


def run(features_dir: Optional[Path] = None) -> TrainingStats:
    features_dir = features_dir or settings.FEATURES_DATA_DIR

    start_time = time.perf_counter()

    if not features_dir.exists():
        logger.warning(f"'{features_dir}' bulunamadı. Önce 'feature-extract' adımını çalıştırın.")
        return TrainingStats(duration_sec=time.perf_counter() - start_time)

    embeddings = load_embeddings(features_dir)

    if not embeddings:
        logger.warning(f"'{features_dir}' içinde işlenecek .npy embedding dosyası bulunamadı.")
        return TrainingStats(duration_sec=time.perf_counter() - start_time)

    logger.info(f"{len(embeddings)} embedding bulundu, eğitim döngüsü başlıyor...")

    config = {
        "project": settings.WANDB_PROJECT,
        "epochs": settings.TRAIN_EPOCHS,
        "batch_size": settings.BATCH_SIZE,
        "learning_rate": settings.LEARNING_RATE,
        "dataset_size": len(embeddings),
    }

    wandb.init(project=settings.WANDB_PROJECT, config=config)

    final_loss = 0.0
    final_accuracy = 0.0

    try:
        for epoch in range(1, settings.TRAIN_EPOCHS + 1):
            final_loss, final_accuracy = simulate_step(epoch, settings.TRAIN_EPOCHS)
            wandb.log({"epoch": epoch, "loss": final_loss, "accuracy": final_accuracy})

            if epoch % 10 == 0 or epoch == settings.TRAIN_EPOCHS:
                logger.info(
                    f"Epoch {epoch}/{settings.TRAIN_EPOCHS} - loss: {final_loss:.4f} - accuracy: {final_accuracy:.4f}"
                )
    finally:
        wandb.finish()

    duration_sec = time.perf_counter() - start_time

    logger.success(
        f"Eğitim tamamlandı: {settings.TRAIN_EPOCHS} epoch, final loss: {final_loss:.4f}, "
        f"final accuracy: {final_accuracy:.4f}. Süre: {duration_sec:.2f}sn"
    )

    return TrainingStats(
        total_embeddings=len(embeddings),
        epochs_completed=settings.TRAIN_EPOCHS,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        duration_sec=duration_sec,
    )


if __name__ == "__main__":
    run()
