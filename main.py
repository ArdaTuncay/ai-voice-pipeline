import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.features import extractor
from src.models import inference as inference_module
from src.models import train as train_module
from src.preprocessor import cleaner
from src.utils.config import settings
from src.utils.logger_manager import logger

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = typer.Typer(
    name="ai-voice-pipeline",
    help="RVC v2 tabanlı AI Voice Engineering Pipeline CLI.",
    add_completion=False,
)

console = Console()


@app.callback()
def main() -> None:
    """RVC v2 tabanlı AI Voice Engineering Pipeline — modüler komut satırı arayüzü."""


@app.command()
def preprocess(
    raw_dir: Optional[Path] = typer.Option(
        None,
        "--raw-dir",
        "-r",
        help="Ham ses dosyalarının bulunduğu dizin. Belirtilmezse PipelineSettings.RAW_DATA_DIR kullanılır.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="İşlenmiş dosyaların kaydedileceği dizin. Belirtilmezse PipelineSettings.PROCESSED_DATA_DIR kullanılır.",
    ),
) -> None:
    """Ham sesleri temizler, normalize eder, parçalar ve işlenmiş dizine kaydeder."""
    effective_raw_dir = raw_dir or settings.RAW_DATA_DIR
    effective_output_dir = output_dir or settings.PROCESSED_DATA_DIR

    logger.info("=" * 60)
    logger.info("preprocess komutu başlatıldı")
    logger.info(f"Raw dir: {effective_raw_dir} | Output dir: {effective_output_dir}")

    console.print(f"[bold cyan]▶ Ön işleme başlatılıyor...[/bold cyan] ({effective_raw_dir} → {effective_output_dir})")

    try:
        stats = cleaner.run(raw_dir=effective_raw_dir, processed_dir=effective_output_dir)
    except Exception as e:
        logger.exception(f"preprocess komutu beklenmeyen bir hata ile sonlandı: {e}")
        console.print(f"[bold red]✗ Ön işleme başarısız oldu:[/bold red] {e}")
        raise typer.Exit(code=1)

    logger.info("preprocess komutu tamamlandı")

    table = Table(title="Ön İşleme Sonucu", show_header=True, header_style="bold cyan")
    table.add_column("Metrik")
    table.add_column("Değer", justify="right")
    table.add_row("Taranan dosya", str(stats.total_files))
    table.add_row("Oluşturulan parça", str(stats.total_chunks))
    table.add_row("Başarısız dosya", str(stats.failed_files))
    table.add_row("Örnekleme hızı", f"{settings.SAMPLE_RATE} Hz")
    table.add_row("Maks. parça süresi", f"{settings.MAX_CHUNK_DURATION_SEC} sn")

    console.print(table)

    if stats.total_files == 0:
        console.print(f"[bold yellow]⚠ '{effective_raw_dir}' içinde işlenecek dosya bulunamadı.[/bold yellow]")
    elif stats.failed_files > 0:
        console.print(f"[bold yellow]⚠ Tamamlandı, ancak {stats.failed_files} dosya başarısız oldu.[/bold yellow]")
    else:
        console.print("[bold green]✓ Ön işleme başarıyla tamamlandı.[/bold green]")


@app.command(name="feature-extract")
def feature_extract(
    processed_dir: Optional[Path] = typer.Option(
        None,
        "--processed-dir",
        "-p",
        help="İşlenmiş .wav dosyalarının bulunduğu dizin. Belirtilmezse PipelineSettings.PROCESSED_DATA_DIR kullanılır.",
    ),
    features_dir: Optional[Path] = typer.Option(
        None,
        "--features-dir",
        "-f",
        help="Embedding (.npy) dosyalarının kaydedileceği dizin. Belirtilmezse PipelineSettings.FEATURES_DATA_DIR kullanılır.",
    ),
) -> None:
    """İşlenmiş ses parçalarından resemblyzer ile speaker embedding (d-vector) çıkarır."""
    effective_processed_dir = processed_dir or settings.PROCESSED_DATA_DIR
    effective_features_dir = features_dir or settings.FEATURES_DATA_DIR

    logger.info("=" * 60)
    logger.info("feature-extract komutu başlatıldı")
    logger.info(f"Processed dir: {effective_processed_dir} | Features dir: {effective_features_dir}")

    console.print(
        f"[bold cyan]▶ Özellik çıkarımı başlatılıyor...[/bold cyan] "
        f"({effective_processed_dir} → {effective_features_dir})"
    )

    try:
        stats = extractor.run(processed_dir=effective_processed_dir, features_dir=effective_features_dir)
    except Exception as e:
        logger.exception(f"feature-extract komutu beklenmeyen bir hata ile sonlandı: {e}")
        console.print(f"[bold red]✗ Özellik çıkarımı başarısız oldu:[/bold red] {e}")
        raise typer.Exit(code=1)

    logger.info("feature-extract komutu tamamlandı")

    table = Table(title="Özellik Çıkarımı Sonucu", show_header=True, header_style="bold cyan")
    table.add_column("Metrik")
    table.add_column("Değer", justify="right")
    table.add_row("Taranan dosya", str(stats.total_files))
    table.add_row("Başarılı embedding", str(stats.succeeded))
    table.add_row("Başarısız dosya", str(stats.failed))
    table.add_row("Süre", f"{stats.duration_sec:.2f} sn")

    console.print(table)

    if stats.total_files == 0:
        console.print(
            f"[bold yellow]⚠ '{effective_processed_dir}' içinde işlenecek .wav dosyası bulunamadı.[/bold yellow]"
        )
    elif stats.failed > 0:
        console.print(f"[bold yellow]⚠ Tamamlandı, ancak {stats.failed} dosya başarısız oldu.[/bold yellow]")
    else:
        console.print("[bold green]✓ Özellik çıkarımı başarıyla tamamlandı.[/bold green]")


@app.command()
def train(
    features_dir: Optional[Path] = typer.Option(
        None,
        "--features-dir",
        "-f",
        help="Embedding (.npy) dosyalarının bulunduğu dizin. Belirtilmezse PipelineSettings.FEATURES_DATA_DIR kullanılır.",
    ),
) -> None:
    """Embedding'ler üzerinde eğitim döngüsünü çalıştırır ve metrikleri W&B'ye loglar."""
    effective_features_dir = features_dir or settings.FEATURES_DATA_DIR

    logger.info("=" * 60)
    logger.info("train komutu başlatıldı")
    logger.info(f"Features dir: {effective_features_dir}")

    console.print(f"[bold cyan]▶ Eğitim başlatılıyor...[/bold cyan] ({effective_features_dir})")

    try:
        stats = train_module.run(features_dir=effective_features_dir)
    except Exception as e:
        logger.exception(f"train komutu beklenmeyen bir hata ile sonlandı: {e}")
        console.print(f"[bold red]✗ Eğitim başarısız oldu:[/bold red] {e}")
        raise typer.Exit(code=1)

    logger.info("train komutu tamamlandı")

    table = Table(title="Eğitim Özeti", show_header=True, header_style="bold cyan")
    table.add_column("Metrik")
    table.add_column("Değer", justify="right")
    table.add_row("Kullanılan embedding", str(stats.total_embeddings))
    table.add_row("Tamamlanan epoch", str(stats.epochs_completed))
    table.add_row("Final loss", f"{stats.final_loss:.4f}")
    table.add_row("Final accuracy", f"{stats.final_accuracy:.4f}")
    table.add_row("Batch size", str(settings.BATCH_SIZE))
    table.add_row("Learning rate", str(settings.LEARNING_RATE))
    table.add_row("W&B projesi", settings.WANDB_PROJECT)
    table.add_row("Süre", f"{stats.duration_sec:.2f} sn")

    console.print(table)

    if stats.total_embeddings == 0:
        console.print(
            f"[bold yellow]⚠ '{effective_features_dir}' içinde işlenecek .npy dosyası bulunamadı.[/bold yellow]"
        )
    else:
        console.print("[bold green]✓ Eğitim başarıyla tamamlandı.[/bold green]")


@app.command()
def inference(
    source_audio: Path = typer.Option(
        ...,
        "--source-audio",
        "-s",
        help="Dönüştürülecek kaynak ses dosyası.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output-path",
        "-o",
        help="Dönüştürülmüş sesin kaydedileceği dosya yolu. Belirtilmezse PipelineSettings.OUTPUT_DATA_DIR altına otomatik isimlendirilir.",
    ),
) -> None:
    """Kaynak sesi, çıkarılan hedef konuşmacı tınısına dönüştürüp data/output/ altına kaydeder."""
    logger.info("=" * 60)
    logger.info("inference komutu başlatıldı")
    logger.info(f"Source audio: {source_audio} | Output path: {output_path or '(otomatik)'}")

    console.print(f"[bold cyan]▶ Ses dönüşümü başlatılıyor...[/bold cyan] ({source_audio})")

    try:
        result = inference_module.run(source_audio=source_audio, output_path=output_path)
    except Exception as e:
        logger.exception(f"inference komutu beklenmeyen bir hata ile sonlandı: {e}")
        console.print(f"[bold red]✗ Ses dönüşümü başarısız oldu:[/bold red] {e}")
        raise typer.Exit(code=1)

    logger.info("inference komutu tamamlandı")

    if not result.success:
        console.print("[bold red]✗ Ses dönüşümü başarısız oldu.[/bold red] Ayrıntılar için loglara bakın.")
        raise typer.Exit(code=1)

    table = Table(title="Ses Dönüşümü Sonucu", show_header=True, header_style="bold cyan")
    table.add_column("Metrik")
    table.add_column("Değer", justify="right")
    table.add_row("Çıktı konumu", str(result.output_path))
    table.add_row("Ses süresi", f"{result.duration_sec:.2f} sn")
    table.add_row("İşlem süresi", f"{result.processing_time_sec:.2f} sn")
    table.add_row("Pitch kayması", f"{result.pitch_shift_semitones:+.2f} yarım ton")
    table.add_row("Formant oranı", f"x{result.formant_shift_ratio:.3f}")

    console.print(table)
    console.print("[bold green]✓ Ses dönüşümü başarıyla tamamlandı.[/bold green]")


if __name__ == "__main__":
    app()
