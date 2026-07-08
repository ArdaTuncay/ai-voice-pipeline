import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyworld as pw
import soundfile as sf
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from src.utils.config import settings
from src.utils.ffmpeg_setup import ensure_ffmpeg_on_path
from src.utils.logger_manager import logger

ensure_ffmpeg_on_path()

FRAME_PERIOD_MS = 5.0

# Aşırı bükülmeyi (artefakt/robotikleşmeyi) önlemek için ölçülen oranlara uygulanan
# güvenlik sınırları: pitch en fazla ±1 oktav, formant en fazla ~%35 kayabilir.
PITCH_RATIO_MIN = 0.5
PITCH_RATIO_MAX = 2.0
FORMANT_RATIO_MIN = 0.75
FORMANT_RATIO_MAX = 1.35


@dataclass
class InferenceResult:
    success: bool = False
    output_path: Optional[Path] = None
    duration_sec: float = 0.0
    processing_time_sec: float = 0.0
    pitch_shift_semitones: float = 0.0
    formant_shift_ratio: float = 1.0


@dataclass
class VoiceProfile:
    """Bir konuşmacının gerçek kayıtlarından ölçülen ortalama akustik parmak izi."""

    mean_log_f0: float
    spectral_centroid: float
    fs: int


def _load_mono_float64(file_path: Path, target_sr: int) -> np.ndarray:
    """Herhangi bir ses dosyasını (ffmpeg üzerinden) tekil kanal, hedef örnekleme
    hızında, WORLD'ün beklediği [-1, 1] aralığında float64 dalga formuna çevirir."""
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float64) / 32768.0
    return np.ascontiguousarray(samples)


def _analyze(wav: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WORLD vocoder ile F0 konturu, spektral zarf (tını) ve aperiyodiklik çıkarır."""
    f0, t = pw.dio(wav, fs, frame_period=FRAME_PERIOD_MS)
    f0 = pw.stonemask(wav, f0, t, fs)
    sp = pw.cheaptrick(wav, f0, t, fs)
    ap = pw.d4c(wav, f0, t, fs)
    return f0, sp, ap


def _spectral_centroid(freqs: np.ndarray, envelope: np.ndarray) -> float:
    total = float(envelope.sum())
    if total <= 0:
        return float(freqs.mean())
    return float(np.sum(freqs * envelope) / total)


def _warp_spectral_envelope(sp: np.ndarray, warp_ratio: float, fs: int) -> np.ndarray:
    """Spektral zarfı frekans ekseninde ölçekleyerek formantları kaydırır
    (basit frekans-warping; tam formant/LPC izleme yerine geçen hafif bir teknik).

    warp_ratio > 1  -> formantlar yukarı kayar (daha 'ince'/parlak tını)
    warp_ratio < 1  -> formantlar aşağı kayar (daha 'kalın'/koyu tını)
    """
    if abs(warp_ratio - 1.0) < 1e-6:
        return sp

    n_frames, n_bins = sp.shape
    freqs = np.linspace(0.0, fs / 2.0, n_bins)
    src_freqs = freqs / warp_ratio

    warped = np.empty_like(sp)
    for i in range(n_frames):
        warped[i] = np.interp(src_freqs, freqs, sp[i], left=sp[i, 0], right=sp[i, -1])
    return warped


def _has_target_embeddings(features_dir: Path) -> bool:
    if not features_dir.exists():
        return False
    return any(f.is_file() and f.suffix.lower() == ".npy" for f in features_dir.iterdir())


def compute_target_voice_profile(processed_dir: Path, target_sr: int) -> Optional[VoiceProfile]:
    """Hedef konuşmacının işlenmiş kayıtlarından (data/processed/*.wav) gerçek
    ortalama pitch (F0) ve spektral ağırlık merkezi istatistiklerini çıkarır.

    Not: resemblyzer d-vector'ları (data/features/*.npy) soyut konuşmacı
    doğrulama vektörleridir, frekans/formant değerlerine geri çözülemezler.
    Bu yüzden gerçek akustik istatistikler için embedding'lerin üretildiği
    ham kayıtlar (processed_dir) taranır.
    """
    if not processed_dir.exists():
        return None

    wav_files = sorted(
        f for f in processed_dir.iterdir() if f.is_file() and f.suffix.lower() == ".wav"
    )
    if not wav_files:
        return None

    log_f0_samples = []
    envelope_frames = []

    for file_path in wav_files:
        try:
            wav = _load_mono_float64(file_path, target_sr)
        except (CouldntDecodeError, FileNotFoundError, OSError) as e:
            logger.warning(f"Hedef profil için ses okunamadı: {file_path.name} - {e}")
            continue

        f0, sp, _ap = _analyze(wav, target_sr)
        voiced = f0 > 0
        if not np.any(voiced):
            continue

        log_f0_samples.append(np.log(f0[voiced]))
        envelope_frames.append(sp[voiced])

    if not log_f0_samples:
        return None

    mean_log_f0 = float(np.concatenate(log_f0_samples).mean())
    avg_envelope = np.concatenate(envelope_frames, axis=0).mean(axis=0)
    freqs = np.linspace(0.0, target_sr / 2.0, avg_envelope.shape[0])
    centroid = _spectral_centroid(freqs, avg_envelope)

    return VoiceProfile(mean_log_f0=mean_log_f0, spectral_centroid=centroid, fs=target_sr)


def run(
    source_audio: Path,
    output_path: Optional[Path] = None,
    features_dir: Optional[Path] = None,
) -> InferenceResult:
    features_dir = features_dir or settings.FEATURES_DATA_DIR
    processed_dir = settings.PROCESSED_DATA_DIR
    output_dir = settings.OUTPUT_DATA_DIR
    target_sr = settings.SAMPLE_RATE

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
        effective_output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Çıktı dizini oluşturulamadı: {e}")
        return InferenceResult(duration_sec=time.perf_counter() - start_time)

    if not _has_target_embeddings(features_dir):
        logger.warning(
            f"'{features_dir}' içinde embedding bulunamadı; önce 'feature-extract' adımı çalıştırılmalı."
        )

    try:
        source_wav = _load_mono_float64(source_audio, target_sr)
    except (CouldntDecodeError, FileNotFoundError, OSError) as e:
        logger.error(f"Kaynak ses okunamadı: {source_audio.name} - {e}")
        return InferenceResult(duration_sec=time.perf_counter() - start_time)

    target_profile = compute_target_voice_profile(processed_dir, target_sr)
    if target_profile is None:
        logger.warning(
            f"'{processed_dir}' içinde hedef konuşmacı kaydı bulunamadı; "
            "pitch/formant dönüşümü uygulanmadan ses olduğu gibi yazılıyor."
        )
        try:
            sf.write(str(effective_output_path), source_wav, target_sr, subtype="PCM_16")
        except (OSError, PermissionError) as e:
            logger.error(f"Çıktı kaydedilemedi: {effective_output_path} - {e}")
            return InferenceResult(duration_sec=time.perf_counter() - start_time)

        processing_time_sec = time.perf_counter() - start_time
        return InferenceResult(
            success=True,
            output_path=effective_output_path,
            duration_sec=len(source_wav) / target_sr,
            processing_time_sec=processing_time_sec,
        )

    logger.info(f"Kaynak ses analiz ediliyor (F0/spektral zarf çıkarımı): {source_audio.name}")
    f0, sp, ap = _analyze(source_wav, target_sr)
    voiced = f0 > 0

    if not np.any(voiced):
        logger.warning("Kaynak seste sesli (voiced) çerçeve bulunamadı, pitch/formant dönüşümü atlanıyor.")
        pitch_ratio = 1.0
        formant_ratio = 1.0
    else:
        source_mean_log_f0 = float(np.log(f0[voiced]).mean())
        freqs = np.linspace(0.0, target_sr / 2.0, sp.shape[1])
        source_avg_envelope = sp[voiced].mean(axis=0)
        source_centroid = _spectral_centroid(freqs, source_avg_envelope)

        pitch_ratio = float(
            np.clip(
                math.exp(target_profile.mean_log_f0 - source_mean_log_f0),
                PITCH_RATIO_MIN,
                PITCH_RATIO_MAX,
            )
        )
        formant_ratio = float(
            np.clip(
                target_profile.spectral_centroid / source_centroid if source_centroid > 0 else 1.0,
                FORMANT_RATIO_MIN,
                FORMANT_RATIO_MAX,
            )
        )

    semitone_shift = 12.0 * math.log2(pitch_ratio)
    logger.info(
        f"Ölçülen dönüşüm: pitch oranı x{pitch_ratio:.3f} ({semitone_shift:+.2f} yarım ton), "
        f"formant oranı x{formant_ratio:.3f}"
    )

    f0_shifted = f0.copy()
    f0_shifted[voiced] *= pitch_ratio
    sp_shifted = _warp_spectral_envelope(sp, formant_ratio, target_sr)

    try:
        converted = pw.synthesize(f0_shifted, sp_shifted, ap, target_sr, FRAME_PERIOD_MS)
    except Exception as e:
        logger.error(f"WORLD yeniden sentezleme hatası: {e}")
        return InferenceResult(duration_sec=time.perf_counter() - start_time)

    peak = float(np.max(np.abs(converted))) if converted.size else 0.0
    if peak > 0.99:
        converted = converted / peak * 0.99

    try:
        sf.write(str(effective_output_path), converted.astype(np.float32), target_sr, subtype="PCM_16")
    except (OSError, PermissionError) as e:
        logger.error(f"Çıktı kaydedilemedi: {effective_output_path} - {e}")
        return InferenceResult(duration_sec=time.perf_counter() - start_time)

    processing_time_sec = time.perf_counter() - start_time
    output_duration_sec = len(converted) / target_sr

    logger.success(
        f"Dönüşüm tamamlandı: {effective_output_path} ({output_duration_sec:.2f} sn). "
        f"İşlem süresi: {processing_time_sec:.2f}sn"
    )

    return InferenceResult(
        success=True,
        output_path=effective_output_path,
        duration_sec=output_duration_sec,
        processing_time_sec=processing_time_sec,
        pitch_shift_semitones=semitone_shift,
        formant_shift_ratio=formant_ratio,
    )
