import glob
import os
import shutil

from pydub import AudioSegment

from src.utils.logger_manager import logger

_configured = False


def _resolve_binary(name: str) -> str | None:
    """PATH'te bulunamayan ffmpeg/ffprobe için bilinen Windows kurulum yollarına düşer.

    shutil.which, winget'in App Execution Alias'ları yüzünden (veya PATH
    terminal yeniden başlatılana kadar güncellenmediği için) sık sık boş
    dönebiliyor; bu yüzden yaygın kurulum dizinlerini de tarıyoruz.
    """
    found = shutil.which(name)
    if found:
        return found

    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", "C:/Program Files")

    candidate_patterns = [
        f"{program_files}/ffmpeg/bin/{name}.exe",
        f"C:/ffmpeg/bin/{name}.exe",
        f"{local_app_data}/Microsoft/WinGet/Links/{name}.exe",
        f"{local_app_data}/Microsoft/WinGet/Packages/*Gyan.FFmpeg*/*/bin/{name}.exe",
        f"{local_app_data}/Microsoft/WinGet/Packages/*ffmpeg*/*/bin/{name}.exe",
    ]

    for pattern in candidate_patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return None


def ensure_ffmpeg_on_path() -> None:
    """ffmpeg'i AudioSegment.converter'a, ffprobe'u process PATH'ine bağlar.

    Fikir birliği tek seferliktir (modül seviyesinde _configured bayrağı ile
    korunur) ve import sırasından bağımsız olarak her çağıran modülün
    (cleaner.py, inference.py, ...) kendi başına güvenle çağırabilmesi için
    idempotent tasarlandı.
    """
    global _configured
    if _configured:
        return
    _configured = True

    ffmpeg_path = _resolve_binary("ffmpeg")
    ffprobe_path = _resolve_binary("ffprobe")

    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
    else:
        logger.warning("ffmpeg bulunamadı; pydub varsayılan PATH çözümlemesine güvenecek.")

    # NOT: pydub'da "AudioSegment.ffprobe" diye ayarlanabilir bir attribute yok.
    # ffprobe araması pydub.utils.get_prober_name() -> which() üzerinden, her
    # çağrıda doğrudan os.environ["PATH"] taranarak yapılıyor (audio_segment.py'deki
    # .converter/.ffmpeg attribute'u yalnızca dönüştürme/export adımını etkiliyor,
    # probe adımını etkilemiyor). Bu yüzden ffprobe'u devreye almanın tek yolu,
    # bulunduğu dizini process PATH'ine eklemek.
    if ffprobe_path:
        ffprobe_dir = os.path.dirname(ffprobe_path)
        if ffprobe_dir not in os.environ["PATH"].split(os.pathsep):
            os.environ["PATH"] = ffprobe_dir + os.pathsep + os.environ["PATH"]
    else:
        logger.warning("ffprobe bulunamadı; pydub varsayılan PATH çözümlemesine güvenecek.")
