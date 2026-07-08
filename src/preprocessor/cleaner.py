from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydub.silence import split_on_silence

from src.utils.config import settings
from src.utils.ffmpeg_setup import ensure_ffmpeg_on_path
from src.utils.logger_manager import logger

ensure_ffmpeg_on_path()

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
TARGET_DBFS = -20.0
KEEP_SILENCE_MS = 100


@dataclass
class PreprocessStats:
    total_files: int = 0
    total_chunks: int = 0
    failed_files: int = 0


def normalize_audio(audio: AudioSegment, target_dbfs: float = TARGET_DBFS) -> AudioSegment:
    if audio.dBFS == float("-inf"):
        logger.warning("Ses tamamen sessiz, normalizasyon atlanıyor.")
        return audio
    change_in_dbfs = target_dbfs - audio.dBFS
    return audio.apply_gain(change_in_dbfs)


def remove_silence(audio: AudioSegment) -> AudioSegment:
    chunks = split_on_silence(
        audio,
        min_silence_len=settings.MIN_SILENCE_LEN_MS,
        silence_thresh=settings.SILENCE_THRESH_DB,
        keep_silence=KEEP_SILENCE_MS,
    )

    if not chunks:
        logger.warning("Sessizlik temizleme sonrası ses parçası kalmadı, orijinal ses korunuyor.")
        return audio

    combined = AudioSegment.empty()
    for chunk in chunks:
        combined += chunk
    return combined


def split_into_chunks(audio: AudioSegment) -> list[AudioSegment]:
    max_len_ms = settings.MAX_CHUNK_DURATION_SEC * 1000
    if max_len_ms <= 0:
        logger.error("MAX_CHUNK_DURATION_SEC 0 veya negatif olamaz, parçalama atlanıyor.")
        return [audio]

    chunks = [audio[i : i + max_len_ms] for i in range(0, len(audio), max_len_ms)]
    return chunks or [audio]


def process_file(file_path: Path, processed_dir: Path) -> int:
    logger.info(f"İşleniyor: {file_path.name}")

    try:
        audio = AudioSegment.from_file(file_path)
    except (CouldntDecodeError, FileNotFoundError, OSError) as e:
        logger.error(f"Ses dosyası okunamadı: {file_path.name} - {e}")
        return 0

    try:
        audio = audio.set_frame_rate(settings.SAMPLE_RATE).set_channels(1).set_sample_width(2)
        audio = remove_silence(audio)
        audio = normalize_audio(audio)
    except Exception as e:
        logger.error(f"Ön işleme sırasında hata: {file_path.name} - {e}")
        return 0

    if len(audio) == 0:
        logger.warning(f"İşlem sonrası boş ses, atlanıyor: {file_path.name}")
        return 0

    chunks = split_into_chunks(audio)
    saved_count = 0

    for idx, chunk in enumerate(chunks):
        out_path = processed_dir / f"{file_path.stem}_part{idx:03d}.wav"
        try:
            chunk.export(out_path, format="wav")
            saved_count += 1
        except (OSError, PermissionError) as e:
            logger.error(f"Kaydetme hatası: {out_path.name} - {e}")

    logger.success(f"{file_path.name} -> {saved_count} parça oluşturuldu.")
    return saved_count


def run(raw_dir: Path | None = None, processed_dir: Path | None = None) -> PreprocessStats:
    raw_dir = raw_dir or settings.RAW_DATA_DIR
    processed_dir = processed_dir or settings.PROCESSED_DATA_DIR

    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Dizinler oluşturulamadı: {e}")
        return PreprocessStats()

    files = sorted(
        f for f in raw_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning(f"'{raw_dir}' içinde işlenecek ses dosyası bulunamadı.")
        return PreprocessStats()

    logger.info(f"{len(files)} ses dosyası bulundu, temizleme işlemi başlıyor...")

    total_chunks = 0
    failed_files = 0
    for file_path in files:
        chunks_saved = process_file(file_path, processed_dir)
        if chunks_saved == 0:
            failed_files += 1
        total_chunks += chunks_saved

    logger.success(
        f"Tamamlandı: {len(files)} dosya tarandı, {total_chunks} parça kaydedildi, "
        f"{failed_files} dosya başarısız oldu."
    )

    return PreprocessStats(total_files=len(files), total_chunks=total_chunks, failed_files=failed_files)


if __name__ == "__main__":
    run()
