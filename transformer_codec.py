import torch, torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from pathlib import Path

import soundfile as sf


audio_dirs = [
    Path("/Users/jinglis/Desktop/Desktop - James’s MacBook Pro/packs/dave-parkinson-trance-essentials-vol-2-full-pack/DPTE2 Drum Loops"),
    Path("/Users/jinglis/Desktop/Desktop - James’s MacBook Pro/packs/Freshly Squeezed Samples - Dave Parkinson Trance Essentials/DPTE Drum Loops/DPTE Drum Loops - 140 BPM"),
    Path("/Users/jinglis/Documents/PhD/Deep Learning for Audio and Music/generative_audio/DPTE Drum Loops - 140 BPM - converted")
]

num_files = sum(1 for p in Path("/Users/jinglis/Documents/PhD/Deep Learning for Audio and Music/generative_audio/DPTE Drum Loops - 140 BPM - converted").iterdir() if p.is_file())

print(num_files)
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

        if round(duration, 2) == 3.43:
            two_bar_files.append(wav_file)

    two_bar_files_by_dir.append(two_bar_files)

    print(f"\n{audio_dir.name}: {len(two_bar_files)} two-bar loops")

audio_paths = two_bar_files_by_dir[0] + two_bar_files_by_dir[1] + two_bar_files_by_dir[2]

codec = EncodecModel.encodec_model_24khz()
codec.set_target_bandwidth(6.0)       # same setting you tested: 8 codebooks, 75 frames/s
codec.eval()

K, VOCAB, PAD = 3, 1025, 1024          # 1024 real codes + one padding id

import math

FPS = 75.0                                   # EnCodec 24 kHz frame rate
BPM = 140.0
FRAMES_PER_BEAT = FPS * 60.0 / BPM           # ~32.14

def beat_features(L):
    t = torch.arange(L, dtype=torch.float32)                    # step index
    beats = t / FRAMES_PER_BEAT                                  # position measured in beats
    periods = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])      # 16th, 8th, beat, 2 beats, bar, loop
    angles = 2 * math.pi * beats.unsqueeze(1) / periods          # (L, 6)
    return torch.cat([angles.sin(), angles.cos()], dim=1)        # (L, 12)

def apply_delay(codes):                # (N, K, T) -> (N, K, T+K-1)
    N, K_, T = codes.shape
    out = torch.full((N, K_, T + K_ - 1), PAD, dtype=torch.long)
    for k in range(K_):
        out[:, k, k:k + T] = codes[:, k]
    return out

def undo_delay(d):                     # (K, L) -> (K, T)
    K_, L = d.shape
    T = L - K_ + 1
    return torch.stack([d[k, k:k + T] for k in range(K_)])

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

class TokenTransformer(nn.Module):
    def __init__(self, K=K, vocab=1025, max_len=300, d=384, heads=8, layers=6):
        super().__init__()
        self.K = K
        self.embs = nn.ModuleList([nn.Embedding(vocab, d) for _ in range(K)])
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.register_buffer("beat_feats", beat_features(max_len), persistent=False)       # new
        self.beat_proj = nn.Linear(self.beat_feats.size(1), d)                              # new
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.heads = nn.ModuleList([nn.Linear(d, vocab) for _ in range(K)])

    def forward(self, tokens):                         # tokens: (B, K, L)
        L = tokens.size(2)
        h = sum(self.embs[k](tokens[:, k]) for k in range(self.K)) + self.pos[:, :L]
        h = h + self.pos[:, :L] + self.beat_proj(self.beat_feats[:L]).unsqueeze(0)           # new term
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=tokens.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return torch.stack([head(h) for head in self.heads], dim=1)   # (B, K, L, vocab)
    

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
codes = torch.load("codec_tokens.pt")[:, :K]        
delayed = apply_delay(codes)                                          # (N, K, L)
tgt_all = delayed
inp_all = torch.cat([torch.full_like(delayed[:, :, :1], PAD), delayed[:, :, :-1]], dim=2)   # shift right

ds = TensorDataset(inp_all, tgt_all)
train_set, val_set = random_split(ds, [len(ds) - 100, 100], generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)

model = TokenTransformer().to(device)
epochs = 15
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

def run(loader, train):
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)                                       # (B, K, L, V)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=PAD)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * inp.size(0); n += inp.size(0)
    return total / n

"""
ckpt = torch.load("token_transformer.pth", map_location=device)
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
model.load_state_dict(state)

"""
for epoch in range(1, epochs + 1):
    tr, va = run(train_loader, True), run(val_loader, False)
    sched.step()
    print(f"epoch {epoch}  train {tr:.3f}  val {va:.3f}")
torch.save(model.state_dict(), "token_transformer_cd3_beat.pth")

@torch.no_grad()
def generate(model, T, temperature=0.8, top_k=100):
    model.eval()
    K_ = model.K
    L = T + K_ - 1
    tokens = torch.full((1, K_, L), PAD, dtype=torch.long, device=device)
    for s in range(L):
        inp = torch.cat([torch.full((1, K_, 1), PAD, device=device), tokens[:, :, :s]], dim=2)
        lg = model(inp)[0, :, -1, :PAD] / temperature                  # (K, 1024), pad excluded
        if top_k:
            v, _ = lg.topk(top_k, dim=-1)
            lg[lg < v[:, -1:]] = float("-inf")
        new = torch.multinomial(lg.softmax(-1), 1).squeeze(-1)         # (K,)
        for k in range(K_):
            tokens[0, k, s] = new[k] if k <= s <= k + T - 1 else PAD   # only valid positions get real codes
    return tokens[0].cpu()

delayed_out = generate(model, T=codes.shape[2])

c = undo_delay(delayed_out)                      # (K, T), from your generate call
for n in (1,2):
    with torch.no_grad():
        wav = codec.decode([(c[:n].unsqueeze(0), None)])
    torchaudio.save(f"gen_{n}cb.wav", wav[0], codec.sample_rate)

settings = [(0.6, 100),(1.0, 250), (0.9, 100), (0.8, 50)]    # (temperature, top_k)
n_per_setting = 3
T = codes.shape[2]                                # codes already sliced to K=3

for temp, tk in settings:
    for i in range(n_per_setting):
        torch.manual_seed(i)                      # same seed across settings makes comparison fairer
        d = generate(model, T=T, temperature=temp, top_k=tk)
        c = undo_delay(d)                         # (3, T)
        with torch.no_grad():
            wav = codec.decode([(c.unsqueeze(0), None)])
        torchaudio.save(f"gen_t{temp}_k{tk}_s{i}.wav", wav[0], codec.sample_rate)

def nearest_loop(c, codes):                           # c: (K, T), codes: (N, K, T)
    match = (codes == c.unsqueeze(0)).float().mean(dim=(1, 2))   # fraction of identical tokens
    best = match.max(dim=0)
    return best.values.item(), best.indices.item()

# reference: how similar is a real loop to its closest *other* real loop?
ref = []
for i in range(100):
    m = (codes == codes[i].unsqueeze(0)).float().mean(dim=(1, 2))
    m[i] = 0
    ref.append(m.max().item())
print("typical real-to-real nearest match:", sum(ref) / len(ref))

print("generated sample:", nearest_loop(c, codes))      # compare against the line above
