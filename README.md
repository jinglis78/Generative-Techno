# Generative techno drum loops

Train a convolutional variational autoencoder (VAE) on two-bar techno drum loops. The model learns a latent space of spectrograms and can reconstruct or sample new loops as audio.


## What the code does

1. Collects two-bar `.wav` drum loops (about 3.43 s, i.e. two bars at 140 BPM) from local sample-pack folders.
2. Converts each audio file to a magnitude spectrogram (STFT in dB), caches the tensor, and scales values into `[0, 1]`.
3. Trains `StridedConvVAE`: a 2D convolutional encoder, a compact latent vector, and a transposed-convolution decoder.
4. Saves weights, logs hyperparameters and epoch losses, plots the training curve, and writes a few generated WAVs by inverting spectrograms with Griffin–Lim.

A commented-out block in `main.py` can time-stretch 134–139 BPM loops to 140 BPM (pitch preserved) before training.

## Repository layout

| File | Role |
| --- | --- |
| `main.py` | Dataset selection, spectrogram conversion, training, logging, sampling |
| `vae_model.py` | `StridedConvVAE` (used) and an earlier `GenerativeVAE` |
| `vae_base_class.py` | Training loop and loss (L1 reconstruction + frequency-difference term + KL) |
| `data_processing_class.py` | Load audio, STFT, normalise / denormalise, Griffin–Lim, plot and write files |
| `dpte.py` | Standalone helper that counts two-bar loops in the sample packs |

Generated artefacts (`*.pth`, `output.txt`, spectrogram caches, plots, audio) are gitignored.

## Requirements

- Python 3.10+ recommended
- A machine with enough RAM to hold the spectrogram tensor; Apple Silicon uses MPS if available, otherwise CPU

Install:

```bash
pip install torch torchaudio torchvision librosa soundfile matplotlib scipy numpy ipython
```

On macOS, install the official PyTorch wheels from [pytorch.org](https://pytorch.org) if you want MPS acceleration.

## Data

The sample packs are **not** in this repo. `main.py` currently points at local paths:

- Dave Parkinson Trance Essentials Vol. 2 drum loops
- Dave Parkinson Trance Essentials 140 BPM drum loops
- A converted 140 BPM folder under `generative_audio/`

Folders named `DPTE2 Snare Builds` and `DPTE2 Percussion FX Loops` are skipped.

Edit `audio_dirs` (and the converted-folder path) in `main.py` to match your machine. Training only keeps files whose duration rounds to `3.43` seconds.

Outputs (cached spectrograms, plots, generated audio) go to:

```text
.../generative_audio/DPTE
```

Change `output_dir` in `main.py` if you want them somewhere else. The first spectrogram pass writes `specgrams.pkl` there; later runs reuse that cache if the file still exists.

## How to run

From this directory:

```bash
python main.py
```

`main.py` redirects stdout/stderr to `output.txt` in the same folder.

Default training settings (also logged in `experiment_log.txt`):

- batch size 16, 20 epochs, Adam `lr=0.001`
- latent dim 32
- kernels `(5, 5, 5, 5, 5)`, channels `(32, 64, 128, 256, 256)`
- spectrograms at 22.5 kHz

After training it writes `vae_model_DPTE_vae3.pth` and a few `generated_*.wav` files under the output audio folder.

`dpte.py` is optional: run it to print how many two-bar loops each pack contains, without training.

```bash
python dpte.py
```
