"""
Frame-level causal transformer on magnitude spectrograms.

Each spectrogram frame (513 frequency bins) is treated as one "token" vector.
The model reads frames 0..t and predicts frame t+1 (next-frame prediction, L1 loss).
Audio is recovered from spectrograms with Griffin-Lim.

Run from the repo folder. Set TRAIN below to train from scratch or to load a checkpoint.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as taF
from scipy.io.wavfile import write
from torch.utils.data import DataLoader, TensorDataset, random_split

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
TRAIN = False                              # True: train from scratch. False: load CKPT_PATH.
CKPT_PATH = "frame_transformer.pth"
SPECGRAM_PATH = "./DPTE/specgrams.pkl"     # cache written by main.py (relative to working dir)

EPOCHS = 20
BATCH_SIZE = 16
N_VAL = 100                                # loops held out for validation
SEED_FRAMES = 37                           # ~1 beat at 140 BPM, used to seed generation

# Spectrogram settings; these must match the values used to build specgrams.pkl
# (window ~0.0455 s and hop ~0.0116 s at 22,500 Hz).
SAMPLERATE, WINDOW, HOP = 22500, 1024, 261

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Audio / plotting helpers
# ----------------------------------------------------------------------------
def specgram_to_wav(normed, path, mag=1):
    """Invert a normalised spectrogram to audio with Griffin-Lim and write a 16-bit wav.

    normed: (T, 513) tensor, values in [0, 1] (frames first).
    mag:    extra gain on the output waveform (used to make small residuals audible).
    """
    spec = normed.detach().cpu().float().T            # (513, T): torchaudio wants bins first
    db = spec * 100.0 - 100.0                         # undo (dB + 100) / 100
    magnitude = 10.0 ** (db / 20.0)                   # undo 20 * log10(magnitude)
    waveform = taF.griffinlim(
        magnitude,
        window=torch.hann_window(WINDOW),
        n_fft=WINDOW,
        hop_length=HOP,
        win_length=WINDOW,
        power=1.0,                                    # input is a magnitude, not power
        n_iter=200,
        momentum=0.99,
        length=None,
        rand_init=True,
    )
    waveform = (waveform * 2.0 * mag).numpy()         # x2 gain, plus optional extra gain
    write(path, SAMPLERATE, (np.clip(waveform, -1, 1) * 32767).astype(np.int16))


def save_specgram_comparison(orig, recon, out_path, titles=("original", "reconstruction")):
    """Save a 3-panel figure: original, reconstruction, and their difference.

    orig, recon: (T, 513) tensors of normalised values.
    """
    orig = orig.detach().cpu().numpy().T              # (513, T) for imshow
    recon = recon.detach().cpu().numpy().T
    vmin = min(orig.min(), recon.min())               # shared colour scale for the first two panels
    vmax = max(orig.max(), recon.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, img, title in zip(axes[:2], (orig, recon), titles):
        im = ax.imshow(img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("frames")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("frequency bins")

    diff = orig - recon
    lim = abs(diff).max()                             # symmetric scale so zero is the centre colour
    im = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title("original - reconstruction")
    axes[2].set_xlabel("frames")
    fig.colorbar(im, ax=axes[2])

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class FrameTransformer(nn.Module):
    """Causal transformer: a (B, T, 513) spectrogram in, a prediction of the next frame out."""

    def __init__(self, n_bins=513, n_frames=296, d=256, heads=8, layers=4):
        super().__init__()
        self.inp = nn.Linear(n_bins, d)                                  # frame -> d-dim vector
        self.pos = nn.Parameter(torch.randn(1, n_frames, d) * 0.02)      # learned position per frame index
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d),
                                            enable_nested_tensor=False)
        self.out = nn.Linear(d, n_bins)                                  # d-dim vector -> frame

    def forward(self, x):
        # x: (B, T, 513) normalised frames
        h = self.inp(x) + self.pos[:, : x.size(1)]
        # Causal mask: frame t may only attend to frames <= t.
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        # Output at position t is the prediction of frame t+1, squashed to [0, 1].
        return torch.sigmoid(self.out(h))


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
with open(SPECGRAM_PATH, "rb") as f:
    spec = pickle.load(f)
spec = torch.as_tensor(spec).float()              # cast: may have been stored as int64
print(spec.shape, spec.dtype, spec.min().item(), spec.max().item())    # expect (N, 513, 296)

# dB in [-100, 0] -> [0, 1], then frames first: (N, 296, 513)
x = ((spec + 100.0) / 100.0).transpose(1, 2).contiguous()
print(x.shape)

# Sanity baseline: how well does "repeat the previous frame" do? The model should beat this.
print("copy-previous baseline L1:", F.l1_loss(x[:, :-1], x[:, 1:]).item())

g = torch.Generator().manual_seed(0)
train_set, val_set = random_split(TensorDataset(x), [len(x) - N_VAL, N_VAL], generator=g)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

batch = next(iter(train_loader))[0]
print(batch.shape)                                # expect (16, 296, 513)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
model = FrameTransformer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)


def run(loader, train):
    """One pass over `loader`; returns the mean L1 next-frame loss per loop."""
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for (batch,) in loader:
            batch = batch.to(device)                         # (B, 296, 513)
            pred = model(batch)
            # Prediction at position t is compared with the real frame t+1.
            loss = F.l1_loss(pred[:, :-1], batch[:, 1:])
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * batch.size(0)
            n += batch.size(0)
    return total / n


if TRAIN:
    for epoch in range(1, EPOCHS + 1):
        tr = run(train_loader, True)
        va = run(val_loader, False)
        sched.step()
        print(f"epoch {epoch}  train {tr:.4f}  val {va:.4f}")
    torch.save(model.state_dict(), CKPT_PATH)
else:
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)


# ----------------------------------------------------------------------------
# Teacher-forced reconstruction: the model sees the REAL previous frames at every step
# ----------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    x_sample = val_set[0][0]                                  # one held-out loop, (296, 513)
    pred = model(x_sample.unsqueeze(0).to(device))[0, :-1]    # predictions for frames 1..295

save_specgram_comparison(x_sample[1:], pred, "teacher_forced.png")

specgram_to_wav(x_sample[1:], "orig.wav")
specgram_to_wav(pred, "recon.wav")
# Residual written as audio (gain x400) as a rough diagnostic. It is not a valid spectrogram:
# negative differences map below the -100 dB floor, so they become near-silent.
specgram_to_wav(pred.cpu() - x_sample[1:].cpu(), "recon-orig.wav", 400)


# ----------------------------------------------------------------------------
# Autoregressive generation: the model sees only its OWN previous predictions
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate(model, seed, total_frames=296):
    """Extend `seed` (1, k, 513) to `total_frames` frames by feeding each prediction back in.

    The model is deterministic (L1 regression, no sampling), so the same seed always
    gives the same output.
    """
    model.eval()
    seq = seed.to(device)
    while seq.size(1) < total_frames:
        nxt = model(seq)[:, -1:]                      # prediction for the next frame only
        seq = torch.cat([seq, nxt], dim=1)
    return seq


seed = x_sample[:SEED_FRAMES].unsqueeze(0)            # first ~1 beat of the held-out loop
gen = generate(model, seed)[0]                        # (296, 513)

save_specgram_comparison(x_sample, gen, "autoregressive.png",
                         titles=("original", "generated from 1-beat seed"))

specgram_to_wav(x_sample, "orig_gen.wav")
specgram_to_wav(gen, "recon_gen.wav")
specgram_to_wav(gen.cpu() - x_sample.cpu(), "recon-gen.wav", 400)    # residual diagnostic, as above