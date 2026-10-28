import torch, torchaudio
import csv
from math import ceil
import pickle
import torchaudio.functional as taF
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from IPython.display import display
from collections import Counter

import IPython.display as ipd
from vae_model import StridedConvVAE
from vae_base_class import VAEBaseClass
from data_processing_class import DataProcessor
from pathlib import Path
import sys
import librosa

import soundfile as sf

import os
import inspect
from datetime import datetime

def log_experiment(
    loss_terms,
    loss_function,
    batch_size,
    latent_dim,
    kernel_sizes,
    strides,
    channels,
    losses,
    extra_params=None,
    filename="experiment_log.txt"
):
    separator = "=" * 70

    with open(filename, "a") as f:
        f.write(f"\n{separator}\n")
        f.write(f"NEW MODEL EXPERIMENT: {datetime.now()}\n")
        f.write(f"{separator}\n")

        f.write("\t MODEL PARAMETERS\n")
        f.write("\t ----------------\n")
        f.write(f"\t batch_size = {batch_size}\n")
        f.write(f"\t latent_dim = {latent_dim}\n")
        f.write(f"\t kernel_sizes = {kernel_sizes}\n")
        f.write(f"\t strides = {strides}\n")
        f.write(f"\t channels = {channels}\n")

        f.write("\t ---------------------\n")
        for name, value in (extra_params or {}).items():
            f.write(f"{name} = {value}\n")

        f.write("\t LOSS FUNCTION\n")
        f.write("\t -------------\n")
        try:
            f.write(inspect.getsource(loss_function))
        except (OSError, TypeError):
            f.write("Could not retrieve source automatically.\n")

        f.write("\t EPOCH LOSSES\n")
        f.write("\t ------------\n ")
        for epoch, loss in enumerate(losses, start=1):
            f.write(f"\t Epoch {epoch}: {loss} | 'recon: {loss_terms[epoch-1][0]:.4f}  freq: {loss_terms[epoch-1][1]:.4f}  KL: {loss_terms[epoch-1][2]:.2f} \n")

        f.write(f"{separator}\n\n")

sys.stdout = open("output.txt", "w", buffering=1)
sys.stderr = sys.stdout

# create a list of Path objects, each representing a directory path for a .wav file in the dataset
"""audio_dir = Path('/Users/jinglis/Documents/PhD/Deep Learning for Audio and Music/teaching_material/TinySOL')
with open(audio_dir / 'TinySOL_metadata.csv', "r", newline="") as f:
    csv_file = csv.DictReader(f)
    audio_paths = [audio_dir / row['Path'] for row in csv_file if row['Path']]"""


audio_dirs = [
    Path("/Users/jinglis/Desktop/Desktop - James’s MacBook Pro/packs/dave-parkinson-trance-essentials-vol-2-full-pack/DPTE2 Drum Loops"),
    Path("/Users/jinglis/Desktop/Desktop - James’s MacBook Pro/packs/Freshly Squeezed Samples - Dave Parkinson Trance Essentials/DPTE Drum Loops/DPTE Drum Loops - 140 BPM"),
    Path("/Users/jinglis/Documents/PhD/Deep Learning for Audio and Music/generative_audio/DPTE Drum Loops - 140 BPM - converted")
]

num_files = sum(1 for p in Path("/Users/jinglis/Documents/PhD/Deep Learning for Audio and Music/generative_audio/DPTE Drum Loops - 140 BPM - converted").iterdir() if p.is_file())

print(num_files)
skip_dirs = {"DPTE2 Snare Builds", "DPTE2 Percussion FX Loops"}

two_bar_files_by_dir = []

folder = Path("/Users/jinglis/Documents/PhD/Deep Learning for Audio and Music/generative_audio/DPTE Drum Loops - 140 BPM - converted")

durations = Counter()

for path in folder.glob("*.wav"):
    info = sf.info(str(path))
    duration = info.frames / info.samplerate

    # Round to 2 decimal places to group near-identical durations
    durations[round(duration, 5)] += 1

for duration, count in sorted(durations.items()):
    print(f"{duration:.5f} seconds: {count} files")


""""# The parent containing the 32–39 BPM folders
loops_root = audio_dirs[1].parent

target_bpm = 140
source_bpms = range(134, 140)

# Allow for small differences in file duration
duration_tolerance = 0.03  # 3%

# Save converted files separately
output_folder = Path("/Users/jinglis/Documents/PhD/Deep Learning for Audio and Music/generative_audio/DPTE Drum Loops - 140 BPM - converted")
output_folder.mkdir(parents=True, exist_ok=True)

converted = []
skipped = []

for bpm in source_bpms:
    source_folder = loops_root / f"DPTE Drum Loops - {bpm} BPM"

    if not source_folder.exists():
        print(f"Folder not found: {source_folder}")
        continue

    expected_duration = 8 * 60 / bpm

    wav_files = list(source_folder.rglob("*.wav"))
    wav_files += list(source_folder.rglob("*.WAV"))

    print(f"\n{bpm} BPM: checking {len(wav_files)} files")

    for wav_path in wav_files:
        try:
            info = sf.info(str(wav_path))
            duration = info.frames / info.samplerate

            # Is this approximately two bars long?
            relative_error = abs(duration - expected_duration) / expected_duration

            if relative_error > duration_tolerance:
                skipped.append((wav_path.name, duration, bpm))
                continue

            # Read audio, retaining its original sample rate
            audio, sr = sf.read(str(wav_path), dtype="float32",
                                always_2d=True)

            # librosa expects (channels, samples) for multichannel audio
            audio = audio.T

            # rate > 1 speeds up; preserve the pitch
            rate = target_bpm / bpm
            sped_up = librosa.effects.time_stretch(audio, rate=rate)

            # Return to (samples, channels) for soundfile
            sped_up = sped_up.T

            # Include source BPM in the filename
            output_name = f"{bpm}BPM_{wav_path.stem}.wav"
            output_path = output_folder / output_name

            sf.write(str(output_path), sped_up, sr)

            converted.append(
                (wav_path.name, bpm, duration,
                 len(sped_up) / sr, output_path)
            )

            print(
                f"  Converted: {wav_path.name} "
                f"({duration:.2f}s -> {len(sped_up) / sr:.2f}s)"
            )

        except Exception as exc:
            print(f"  Error processing {wav_path.name}: {exc}")


print("\n========== SUMMARY ==========")
print(f"Converted: {len(converted)}")
print(f"Skipped (not two bars): {len(skipped)}")
print(f"Output folder: {output_folder}")
if skipped:
    print("\nSkipped files:")
    for name, duration, bpm in skipped:
        expected = 8 * 60 / bpm
        print(
            f"  {name}: {duration:.2f}s "
            f"(expected {expected:.2f}s at {bpm} BPM)"
        )

"""



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

        if round(duration, 2) == 3.43:
            two_bar_files.append(wav_file)

    two_bar_files_by_dir.append(two_bar_files)

    print(f"\n{audio_dir.name}: {len(two_bar_files)} two-bar loops")

audio_paths = two_bar_files_by_dir[0] + two_bar_files_by_dir[1] + two_bar_files_by_dir[2]

print(f"Total two-bar loops: {len(audio_paths)}")

info = sf.info(audio_paths[0])

print(f"Sample rate: {info.samplerate} Hz")
print(f"Samples: {info.frames}")
print(f"Exact duration: {info.frames / info.samplerate:.12f} seconds")


# finer resolution spectrograms (e.g. higher sample rate) are inherently unstable -> loss is massive and does not stabaliose
samplerate = 22500
window_duration = 0.0455  # ~91 ms, for n_fft=2048 at 22,500 Hz
hop_duration = 0.01160     # ~11.6 ms, about 261 samples
audio_duration = info.frames / info.samplerate

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

output_dir = "/Users/jinglis/Documents/PhD/Deep Learning for Audio and Music/generative_audio/DPTE"

data = DataProcessor(audio_paths, output_dir, samplerate, window_duration, hop_duration, audio_duration, device)
spec_data = data.convert_to_spectrograms()

norm_data = torch.stack([data.normalise_specgram(specgram) for specgram in spec_data]).unsqueeze(1)


batch_size = 16
data_loader = DataLoader(TensorDataset(norm_data), batch_size, shuffle=True)

input_shape = (1, data.num_bins, data.num_frames)
print ("input_shape: ", input_shape)


print("Using device:", device)

batch_size = 16
latent_dim = 64
kernel_sizes=(5, 5, 5, 5, 5, 5)
strides=((2,2), (2,2), (2,2), (1,1), (1,1),(1,1))
channels=(16, 32, 64, 128, 256, 256)

#model = GenerativeVAE(input_shape=input_shape, latent_dim = 8,device=device)

model = StridedConvVAE(input_shape, 
                       latent_dim, 
                       kernel_sizes=kernel_sizes, 
                       strides=strides, 
                       channels=channels,
                        device=device
                       ).to(device)

model.print_layer_shapes()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainer = VAEBaseClass(model=model, optimizer=optimizer, data_processor=data, device=device)

loss_function = trainer.loss_function

ckpt = torch.load("vae_model_DPTE_vae3.pth", map_location=device)
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
model.load_state_dict(state)

losses, loss_terms = trainer.train_test_sample(model, optimizer, data_loader, data_loader, epochs=20, num_samples=0, latent_dim=latent_dim)
torch.save(model.state_dict(), "vae_model_DPTE_vae3.pth")

print(losses, loss_terms)

extra_params={
        "input_shape": input_shape,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "beta": trainer.beta,
        "sample_rate": samplerate,
        "window_duration": window_duration,
        "hop_duration": hop_duration,
        "epochs": len(losses),
        "dataset_size": len(audio_paths)
    }

log_experiment(
    loss_terms = loss_terms,
    loss_function = loss_function,
    batch_size=batch_size,
    latent_dim=latent_dim,
    kernel_sizes=kernel_sizes,
    strides=strides,
    channels=channels,
    losses=losses, 
    extra_params=extra_params
)

fig, ax1 = plt.subplots()
fig.set_tight_layout(True)
ax1.plot(losses, label='Training')
ax1.set_xlabel('epochs')
ax1.legend()
ax1.set_title('Loss')
fig.savefig(output_dir + "/losses.png", dpi=300, bbox_inches='tight')
plt.close(fig)

"""state_dict = torch.load(
    "vae_model_TinySOL.pth",
    map_location=device,
    weights_only=True
    model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

)"""

data.plot_samples(model, 5, latent_dim)

