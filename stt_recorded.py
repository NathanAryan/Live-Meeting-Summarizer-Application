import os
import shutil
from pathlib import Path
import whisper
from jiwer import wer


def setup_ffmpeg():
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")

    project_root = Path(__file__).resolve().parents[1]  # d:\monty\infosys
    local_ffmpeg = project_root / "ffmpeg" / "bin"
    os.environ["PATH"] = str(local_ffmpeg) + os.pathsep + os.environ.get("PATH", "")
    return shutil.which("ffmpeg")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def main():
    ffmpeg_path = setup_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found. Place it in ffmpeg/bin or add to PATH.")
    print("FFmpeg:", ffmpeg_path)

    base_dir = Path(__file__).resolve().parent
    audio_file = base_dir / "test_audio.wav"
    reference_file = base_dir / "reference.txt"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    model = whisper.load_model("base")
    result = model.transcribe(str(audio_file), word_timestamps=True)

    transcript = result.get("text", "").strip()
    print("\n📄 FULL TEXT:\n", transcript)

    print("\n📌 SEGMENTS:")
    for seg in result.get("segments", []):
        print(f"[{seg.get('start', 0.0):.2f}-{seg.get('end', 0.0):.2f}] 🗣 {seg.get('text', '').strip()}")

    if reference_file.exists():
        ref = normalize_text(reference_file.read_text(encoding="utf-8"))
        hyp = normalize_text(transcript)
        error = wer(ref, hyp)
        print(f"\n📊 WER (test_audio.wav vs reference.txt): {error:.4f}")
    else:
        print("\n⚠️ reference.txt not found (WER skipped)")

    (output_dir / "recorded_transcript.txt").write_text(transcript + "\n", encoding="utf-8")
    print(f"\n✅ Saved: {output_dir / 'recorded_transcript.txt'}")


if __name__ == "__main__":
    main()