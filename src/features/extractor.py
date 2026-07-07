import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from resemblyzer import VoiceEncoder, preprocess_wav

from src.utils.config import settings
from src.utils.logger_manager import logger

_encoder: Optional[VoiceEncoder] = None


@dataclass
class ExtractionStats:
    total_files: int = 0
    succeeded: int = 0
    failed: int = 0
    duration_sec: float = 0.0


def get_encoder() -> VoiceEncoder:
    """VoiceEncoder'ı bir kez oluşturup önbelleğe alır (CUDA varsa GPU, yoksa CPU)."""
    global _encoder
    if _encoder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"VoiceEncoder yükleniyor (device={device})...")
        _encoder = VoiceEncoder(device=device)
    return _encoder


def extract_embedding(file_path: Path, encoder: VoiceEncoder) -> Optional[np.ndarray]:
    try:
        wav = preprocess_wav(file_path)
    except Exception as e:
        logger.error(f"Ses ön işleme (resemblyzer) hatası: {file_path.name} - {e}")
        return None

    if wav.size == 0:
        logger.warning(f"Boş/sessiz ses, embedding çıkarılamadı: {file_path.name}")
        return None

    try:
        return encoder.embed_utterance(wav)
    except Exception as e:
        logger.error(f"Embedding çıkarma hatası: {file_path.name} - {e}")
        return None


def run(processed_dir: Optional[Path] = None, features_dir: Optional[Path] = None) -> ExtractionStats:
    processed_dir = processed_dir or settings.PROCESSED_DATA_DIR
    features_dir = features_dir or settings.FEATURES_DATA_DIR

    start_time = time.perf_counter()

    try:
        features_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Features dizini oluşturulamadı: {e}")
        return ExtractionStats(duration_sec=time.perf_counter() - start_time)

    if not processed_dir.exists():
        logger.warning(f"'{processed_dir}' bulunamadı. Önce 'preprocess' adımını çalıştırın.")
        return ExtractionStats(duration_sec=time.perf_counter() - start_time)

    files = sorted(
        f for f in processed_dir.iterdir() if f.is_file() and f.suffix.lower() == ".wav"
    )

    if not files:
        logger.warning(f"'{processed_dir}' içinde işlenecek .wav dosyası bulunamadı.")
        return ExtractionStats(duration_sec=time.perf_counter() - start_time)

    try:
        encoder = get_encoder()
    except Exception as e:
        logger.error(f"VoiceEncoder yüklenemedi: {e}")
        return ExtractionStats(
            total_files=len(files), failed=len(files), duration_sec=time.perf_counter() - start_time
        )

    logger.info(f"{len(files)} ses dosyası bulundu, embedding çıkarımı başlıyor...")

    succeeded = 0
    failed = 0
    for file_path in files:
        logger.info(f"İşleniyor: {file_path.name}")
        embedding = extract_embedding(file_path, encoder)

        if embedding is None:
            failed += 1
            continue

        out_path = features_dir / f"{file_path.stem}.npy"
        try:
            np.save(out_path, embedding)
            succeeded += 1
            logger.success(f"{file_path.name} -> {out_path.name} ({embedding.shape[0]} boyutlu d-vector)")
        except (OSError, PermissionError) as e:
            logger.error(f"Embedding kaydedilemedi: {out_path.name} - {e}")
            failed += 1

    duration_sec = time.perf_counter() - start_time

    logger.success(
        f"Tamamlandı: {len(files)} dosya tarandı, {succeeded} embedding çıkarıldı, "
        f"{failed} dosya başarısız oldu. Süre: {duration_sec:.2f}sn"
    )

    return ExtractionStats(
        total_files=len(files), succeeded=succeeded, failed=failed, duration_sec=duration_sec
    )


if __name__ == "__main__":
    run()
