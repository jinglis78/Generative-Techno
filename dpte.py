from pathlib import Path
import soundfile as sf
from collections import Counter

from pathlib import Path
import soundfile as sf

audio_dirs = [
    Path("/Users/jinglis/Desktop/Desktop - James’s MacBook Pro/packs/dave-parkinson-trance-essentials-vol-2-full-pack/DPTE2 Drum Loops"),
    Path("/Users/jinglis/Desktop/Desktop - James’s MacBook Pro/packs/Freshly Squeezed Samples - Dave Parkinson Trance Essentials/DPTE Drum Loops/DPTE Drum Loops - 140 BPM"),
]

skip_dirs = {"DPTE2 Snare Builds", "DPTE2 Percussion FX Loops"}

two_bar_files_by_dir = []

for audio_dir in audio_dirs:
    two_bar_files = []

    for wav_file in audio_dir.rglob("*"):
        if not wav_file.is_file() or wav_file.suffix.lower() != ".wav":
            continue

        # Skip files inside excluded directories
        if any(
            part in skip_dirs
            for part in wav_file.relative_to(audio_dir).parts[:-1]
        ):
            continue

        duration = sf.info(wav_file).duration

        if round(duration, 4) == 3.4286:
            two_bar_files.append(wav_file)

    two_bar_files_by_dir.append(two_bar_files)

    print(f"\n{audio_dir.name}: {len(two_bar_files)} two-bar loops")

all_two_bar_files = two_bar_files_by_dir[0] + two_bar_files_by_dir[1]

print(f"Total two-bar loops: {len(all_two_bar_files)}")

info = sf.info(all_two_bar_files[0])

print(f"Sample rate: {info.samplerate} Hz")
print(f"Samples: {info.frames}")
print(f"Exact duration: {info.frames / info.samplerate:.12f} seconds")