import torch
from torch import nn
import torch.nn.functional as F
from data_processing_class import DataProcessor
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

import numpy as np
import torch
import torchaudio.functional as taF
from scipy.io.wavfile import write

SAMPLERATE, WINDOW, HOP = 22500, 1024, 261   # same values your DataProcessor computes

def specgram_to_wav(normed, path, mag = 1):
    # normed: (T, 513) tensor of normalised values (frames first)
    spec = normed.detach().cpu().float().T            # -> (513, T)
    db = spec * 100.0 - 100.0                         # undo (db + 100) / 100
    magnitude = 10.0 ** (db / 20.0)              # undo 20 * log10
    waveform = taF.griffinlim(
        magnitude,
        window=torch.hann_window(WINDOW),
        n_fft=WINDOW,
        hop_length=HOP,
        win_length=WINDOW,
        power=1.0,
        n_iter=200,
        momentum=0.99,
        length=None,
        rand_init=True,
    )
    waveform = (waveform * 2.0 * mag  ).numpy()               # same x2 as your pipeline
    write(path, SAMPLERATE, (np.clip(waveform, -1, 1) * 32767).astype(np.int16))

def save_specgram_comparison(orig, recon, out_path, titles=("original", "reconstruction")):
    # orig, recon: tensors of shape (T, 513), normalised values
    orig = orig.detach().cpu().numpy().T          # -> (513, T)
    recon = recon.detach().cpu().numpy().T
    vmin = min(orig.min(), recon.min())
    vmax = max(orig.max(), recon.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, img, title in zip(axes[:2], (orig, recon), titles):
        im = ax.imshow(img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("frames")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("frequency bins")

    diff = orig - recon
    lim = abs(diff).max()
    im = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title("original - reconstruction")
    axes[2].set_xlabel("frames")
    fig.colorbar(im, ax=axes[2])

    fig.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.close(fig)
    

class FrameTransformer(nn.Module):
    def __init__(self, n_bins=513, n_frames=296, d=256, heads=8, layers=4):
        super().__init__()
        self.inp = nn.Linear(n_bins, d)                         # frame -> vector
        self.pos = nn.Parameter(torch.randn(1, n_frames, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d),
                                    enable_nested_tensor=False)
        self.out = nn.Linear(d, n_bins)                         # vector -> frame                

    def forward(self, x):                   
        # x: (B, T, 513)               
        # turn each frame into a 256d vector and add a learned position vector (one per frame index, 296 in total) is added.
        h = self.inp(x) + self.pos[:, : x.size(1)] 
        # Feed through for attention blocks. 
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        # output: (B, T, 513): prediction of next frame
        return torch.sigmoid(self.out(h))                      

# training step: output at position t predicts frame t+1
# pred = model(x);  target = x.squeeze(1).transpose(1, 2)
# loss = F.l1_loss(pred[:, :-1], target[:, 1:])

import pickle
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open("./DPTE/specgrams.pkl", "rb") as f:
    spec = pickle.load(f)
spec = torch.as_tensor(spec).float()              # cast: torch.full(-100) likely made this int64
print(spec.shape, spec.dtype, spec.min().item(), spec.max().item())   # expect (N, 513, 296)

x = ((spec + 100.0) / 100.0).transpose(1, 2).contiguous()   # (N, 296, 513)
print(x.shape)                                               # now must print (1756, 296, 513)


# baseline: just copy the previous frame
print("copy-previous baseline L1:", F.l1_loss(x[:, :-1], x[:, 1:]).item())

g = torch.Generator().manual_seed(0)
n_val = 100
train_set, val_set = random_split(TensorDataset(x), [len(x) - n_val, n_val], generator=g)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)

batch = next(iter(train_loader))[0]
print(batch.shape)                               # must be (16, 296, 513)

model = FrameTransformer().to(device)
epochs = 20
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

def run(loader, train):
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for (batch,) in loader:
            batch = batch.to(device)                         # (B, 296, 513)
            pred = model(batch)
            loss = F.l1_loss(pred[:, :-1], batch[:, 1:])     # predict the next frame
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * batch.size(0)
            n += batch.size(0)

    return total / n

"""for epoch in range(1, epochs + 1):
    tr = run(train_loader, True)
    va = run(val_loader, False)
    sched.step()
    print(f"epoch {epoch}  train {tr:.4f}  val {va:.4f}")"""

ckpt = torch.load("frame_transformer.pth", map_location=device)
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
model.load_state_dict(state)

model.eval()
with torch.no_grad():
    x_sample = val_set[0][0]                                  # (296, 513)
    pred = model(x_sample.unsqueeze(0).to(device))[0, :-1]    # predictions for frames 1..295

save_specgram_comparison(x_sample[1:], pred, "teacher_forced.png")

specgram_to_wav(x_sample[1:], "orig.wav")
specgram_to_wav(pred, "recon.wav")
specgram_to_wav(pred.cpu()   - x_sample[1:].cpu() , "recon-orig.wav", 400)

data = DataProcessor
torch.save(model.state_dict(), "frame_transformer.pth")

@torch.no_grad()
def generate(model, seed, total_frames=296):          # seed: (1, k, 513)
    model.eval()
    seq = seed.to(device)
    while seq.size(1) < total_frames:
        nxt = model(seq)[:, -1:]                      # predicted next frame
        seq = torch.cat([seq, nxt], dim=1)
    return seq

seed = val_set[0][0][:37].unsqueeze(0)                # first ~1 beat of a held-out loop
out = generate(model, seed)         
gen = generate(model, seed)[0]                   # (1, 296, 513)
                        # (513, 296)

save_specgram_comparison(x_sample, gen, "autoregressive.png",
                         titles=("original", "generated from 1-beat seed"))

specgram_to_wav(x_sample, "orig_gen.wav")
specgram_to_wav(gen, "recon_gen.wav")
specgram_to_wav(gen.cpu()-x_sample.cpu(), "recon-gen.wav",400)