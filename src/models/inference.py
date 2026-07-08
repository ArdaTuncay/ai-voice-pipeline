import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from src.utils.config import settings
from src.utils.logger_manager import logger


@dataclass
class InferenceResult:
    success: bool = False
    output_path: Optional[Path] = None
    duration_sec: float = 0.0
    processing_time_sec: float = 0.0


def load_target_embedding(features_dir: Path) -> Optional[np.ndarray]:
    """Eğitilen modelin yerini tutan hedef konuşmacı tınısını, feature-extract çıktısı
    embedding'lerin ortalaması olarak yükler."""
    if not features_dir.exists():
        return None

    files = sorted(f for f in features_dir.iterdir() if f.is_file() and f.suffix.lower() == ".npy")
    if not files:
        return None

    embeddings = []
    for file_path in files:
        try:
            embeddings.append(np.load(file_path))
        except (OSError, ValueError) as e:
            logger.error(f"Embedding yüklenemedi: {file_path.name} - {e}")

    if not embeddings:
        return None

    return np.mean(embeddings, axis=0)


def apply_voice_conversion(audio: AudioSegment, target_embedding: Optional[np.ndarray]) -> AudioSegment:
    """
    Girdi -> hedef embedding koşullu tını dönüşümü -> çıktı üretim mimarisini kurar.
    Gerçek RVC c++ çekirdeği (nöral pitch/tını dönüşümü) henüz derlenmediğinden,
    çekirdek algoritma burada hedef embedding istatistiklerinden türetilen
    deterministik bir pitch/gain kaymasıyla simüle edilir.
    """
    if target_embedding is None:
        logger.warning("Hedef embedding bulunamadı, ses değişiklik yapılmadan işleniyor.")
        return audio

    embedding_mean = float(np.mean(target_embedding))
    semitone_shift = (embedding_mean % 1.0) * 4.0 - 2.0
    octave_shift = semitone_shift / 12.0

    new_frame_rate = int(audio.frame_rate * (2.0 ** octave_shift))
    converted = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
    converted = converted.set_frame_rate(settings.SAMPLE_RATE)

    gain_shift_db = (embedding_mean % 1.0) * 3.0 - 1.5
    converted = converted.apply_gain(gain_shift_db)

    return converted


def run(
    source_audio: Path,
    output_path: Optional[Path] = None,
    features_dir: Optional[Path] = None,
) -> InferenceResult:
    features_dir = features_dir or settings.FEATURES_DATA_DIR
    output_dir = settings.OUTPUT_DATA_DIR

    start_time = time.perf_counter()

    if not source_audio.exists() or not source_audio.is_file():
        logger.error(f"Kaynak ses dosyası bulunamadı: {source_audio}")
        return InferenceResult(duration_sec=time.perf_counter() - start_time)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Output dizini oluşturulamadı: {e}")
        return InferenceResult(duration_sec=time.perf_counter() - start_time)

    effective_output_path = output_path or (output_dir / f"{source_audio.stem}_converted.wav")

    try:
        audio = AudioSegment.from_file(source_audio)
    except (CouldntDecodeError, FileNotFoundError, OSError) as e:
        logger.error(f"Kaynak ses okunamadı: {source_audio.name} - {e}")
        return InferenceResult(duration_sec=time.perf_counter() - start_time)

    target_embedding = load_target_embedding(features_dir)
    if target_embedding is None:
        logger.warning(f"'{features_dir}' içinde hedef embedding bulunamadı, model eğitilmemiş olabilir.")

    logger.info(f"Ses dönüşümü uygulanıyor: {source_audio.name} -> {effective_output_path.name}")

    try:
        converted = apply_voice_conversion(audio, target_embedding)
    except Exception as e:
        logger.error(f"Ses dönüşümü sırasında hata: {e}")
        return InferenceResult(duration_sec=time.perf_counter() - start_time)

    try:
        converted.export(effective_output_path, format="wav")
    except (OSError, PermissionError) as e:
        logger.error(f"Çıktı kaydedilemedi: {effective_output_path} - {e}")
        return InferenceResult(duration_sec=time.perf_counter() - start_time)

    processing_time_sec = time.perf_counter() - start_time
    output_duration_sec = len(converted) / 1000.0

    logger.success(
        f"Dönüşüm tamamlandı: {effective_output_path} ({output_duration_sec:.2f} sn). "
        f"İşlem süresi: {processing_time_sec:.2f}sn"
    )

    return InferenceResult(
        success=True,
        output_path=effective_output_path,
        duration_sec=output_duration_sec,
        processing_time_sec=processing_time_sec,
    )
