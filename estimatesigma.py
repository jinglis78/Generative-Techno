import torch
from vae_base_class import VAEBaseClass
from vae_model import StridedConvVAE

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
latent_dim = 128
kernel_sizes=(5, 5, 5, 5, 5)
strides=((2,2), (2,2), (2,2), (1,1), (1,1))
channels=(32, 64, 128, 256, 256)

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
model.eval()

@torch.no_grad()
def estimate_sigma(model, loader, device):
    model.eval()
    n, s, s2 = 0, 0, 0
    for data, *_ in loader:
        data = data.to(device)
        _, mu, _ = model.encode(data)
        res = data - model.decode(mu)                 # (B, 1, F, T)
        s  = s  + res.sum(dim=(0, 1, 3))
        s2 = s2 + (res ** 2).sum(dim=(0, 1, 3))
        n += res.shape[0] * res.shape[3]
    mean = s / n
    var = s2 / n - mean ** 2
    return var.clamp_min(0).sqrt().view(1, 1, -1, 1).cpu()   # (1, 1, F, 1)

sigma = estimate_sigma(model, data_loader, device)    # loader = your training DataLoader
print (sigma.size())
torch.save(sigma, "sigma_vae3.pt")               # reuse without recomputing