"""
Causal transformer over EnCodec tokens, with a beat-phase signal for 140 BPM loops.

Pipeline:
  1. Load pre-computed EnCodec tokens (codec_tokens.pt) and keep the first K codebooks.
  2. Apply a "delay pattern" so codebook k is shifted k steps later.
  3. Train a causal transformer to predict the next token for every codebook.
  4. Sample new token sequences, undo the delay, and decode them with EnCodec.
  5. Compare generated samples with the training loops (memorisation check).

Set TRAIN below to train from scratch or to load a saved checkpoint.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from encodec import EncodecModel
from torch.utils.data import DataLoader, TensorDataset, random_split

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
TRAIN = True                                  # False: load CKPT_PATH instead of training
CKPT_PATH = "token_transformer_cd3_beat.pth"
TOKENS_PATH = "codec_tokens.pt"               # pre-computed EnCodec tokens, (N, n_codebooks, T)

K = 3                                         # codebooks used (EnCodec at 6 kbps has 8)
PAD = 1024                                    # padding id, one past the 1024 real codes
VOCAB = 1025                                  # 1024 codes + PAD

EPOCHS = 15
BATCH_SIZE = 16
N_VAL = 100                                   # loops held out for validation

# Beat-phase features: 140 BPM loops, EnCodec 24 kHz runs at 75 token frames per second.
FPS = 75.0
BPM = 140.0
FRAMES_PER_BEAT = FPS * 60.0 / BPM            # ~32.14 token frames per beat

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# EnCodec is only used to turn generated tokens back into audio.
codec = EncodecModel.encodec_model_24khz()
codec.set_target_bandwidth(6.0)               # 6 kbps = 8 codebooks, matching codec_tokens.pt
codec.eval()


# ----------------------------------------------------------------------------
# Beat features and delay pattern
# ----------------------------------------------------------------------------
def beat_features(L):
    """Sin/cos of the position within the beat at several scales -> (L, 12).

    Tells the model where it is in the beat/bar, to help it keep a steady pulse.
    Assumes step 0 is the start of the loop.
    """
    t = torch.arange(L, dtype=torch.float32)                       # step index
    beats = t / FRAMES_PER_BEAT                                    # position measured in beats
    # Periods in beats: 16th, 8th, beat, 2 beats, bar, whole 2-bar loop.
    periods = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    angles = 2 * math.pi * beats.unsqueeze(1) / periods            # (L, 6)
    return torch.cat([angles.sin(), angles.cos()], dim=1)          # (L, 12)


def apply_delay(codes):
    """Shift codebook k later by k steps: (N, K, T) -> (N, K, T + K - 1), padded with PAD.

    With the delay, one time step predicts codebook 0 of the current frame together with
    finer codebooks of earlier frames, so finer codes can condition on coarser ones.
    """
    N, K_, T = codes.shape
    out = torch.full((N, K_, T + K_ - 1), PAD, dtype=torch.long)
    for k in range(K_):
        out[:, k, k:k + T] = codes[:, k]
    return out


def undo_delay(d):
    """Inverse of apply_delay for one sample: (K, L) -> (K, T) with T = L - K + 1."""
    K_, L = d.shape
    T = L - K_ + 1
    return torch.stack([d[k, k:k + T] for k in range(K_)])


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class TokenTransformer(nn.Module):
    """Causal transformer with one embedding table and one output head per codebook."""

    def __init__(self, K=K, vocab=VOCAB, max_len=300, d=384, heads=8, layers=6):
        super().__init__()
        self.K = K
        self.embs = nn.ModuleList([nn.Embedding(vocab, d) for _ in range(K)])   # one per codebook
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)              # learned positions
        # Fixed (non-learned) beat features, projected into the model width.
        self.register_buffer("beat_feats", beat_features(max_len), persistent=False)
        self.beat_proj = nn.Linear(self.beat_feats.size(1), d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.heads = nn.ModuleList([nn.Linear(d, vocab) for _ in range(K)])

    def forward(self, tokens):                            # tokens: (B, K, L)
        L = tokens.size(2)
        # Sum the codebook embeddings so each time step is one vector.
        h = sum(self.embs[k](tokens[:, k]) for k in range(self.K)) + self.pos[:, :L]
        # NOTE: self.pos is added a second time on this line (it is already in the line above).
        # Left as-is so existing checkpoints behave identically; it may have been meant to be
        # `h = h + self.beat_proj(...)` only. Changing it means retraining.
        h = h + self.pos[:, :L] + self.beat_proj(self.beat_feats[:L]).unsqueeze(0)
        # Causal mask: step t may only attend to steps <= t.
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=tokens.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return torch.stack([head(h) for head in self.heads], dim=1)    # (B, K, L, vocab)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
codes = torch.load(TOKENS_PATH)[:, :K]               # (N, n_codebooks, T) -> (N, K, T)
delayed = apply_delay(codes)                         # (N, K, T + K - 1)

# Teacher forcing: inputs are the targets shifted right by one step, starting with PAD.
tgt_all = delayed
inp_all = torch.cat([torch.full_like(delayed[:, :, :1], PAD), delayed[:, :, :-1]], dim=2)

ds = TensorDataset(inp_all, tgt_all)
train_set, val_set = random_split(ds, [len(ds) - N_VAL, N_VAL], generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
model = TokenTransformer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)


def run(loader, train):
    """One pass over `loader`; returns mean cross-entropy over all codebooks (PAD targets ignored)."""
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)                                        # (B, K, L, V)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=PAD)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * inp.size(0)
            n += inp.size(0)
    return total / n


if TRAIN:
    for epoch in range(1, EPOCHS + 1):
        tr, va = run(train_loader, True), run(val_loader, False)
        sched.step()
        print(f"epoch {epoch}  train {tr:.3f}  val {va:.3f}")
    torch.save(model.state_dict(), CKPT_PATH)
else:
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate(model, T, temperature=0.8, top_k=100):
    """Sample a delayed token sequence of length T + K - 1; returns (K, T + K - 1)."""
    model.eval()
    K_ = model.K
    L = T + K_ - 1
    tokens = torch.full((1, K_, L), PAD, dtype=torch.long, device=device)
    for s in range(L):
        # Same input layout as training: PAD first, then everything generated so far.
        inp = torch.cat([torch.full((1, K_, 1), PAD, device=device), tokens[:, :, :s]], dim=2)
        # Logits for step s, one row per codebook; [:PAD] excludes the padding id from sampling.
        lg = model(inp)[0, :, -1, :PAD] / temperature                  # (K, 1024)
        if top_k:
            v, _ = lg.topk(top_k, dim=-1)
            lg[lg < v[:, -1:]] = float("-inf")                         # keep only the top_k logits
        new = torch.multinomial(lg.softmax(-1), 1).squeeze(-1)         # one sampled code per codebook
        for k in range(K_):
            # Codebook k only has real codes at steps k .. k+T-1; elsewhere stay PAD.
            tokens[0, k, s] = new[k] if k <= s <= k + T - 1 else PAD
    return tokens[0].cpu()


def decode_to_wav(c, path):
    """Decode a (n_codebooks, T) token tensor with EnCodec and save it as a wav."""
    with torch.no_grad():
        wav = codec.decode([(c.unsqueeze(0), None)])                   # None: no audio scale factor
    torchaudio.save(path, wav[0], codec.sample_rate)


T = codes.shape[2]                                    # token frames per loop

# One sample at the default settings, decoded with only the first 1 and 2 codebooks.
first_sample = undo_delay(generate(model, T=T))       # (K, T)
for n in (1, 2):
    decode_to_wav(first_sample[:n], f"gen_{n}cb.wav")

# Grid of sampling settings, several seeded samples each, decoded with all K codebooks.
settings = [(0.6, 100), (1.0, 250), (0.9, 100), (0.8, 50)]    # (temperature, top_k)
n_per_setting = 3

for temp, tk in settings:
    for i in range(n_per_setting):
        torch.manual_seed(i)                          # same seeds across settings for fairer comparison
        last_sample = undo_delay(generate(model, T=T, temperature=temp, top_k=tk))   # (K, T)
        decode_to_wav(last_sample, f"gen_t{temp}_k{tk}_s{i}.wav")


# ----------------------------------------------------------------------------
# Memorisation check: does a generated sample just copy a training loop?
# ----------------------------------------------------------------------------
def nearest_loop(c, codes):
    """Best token-match fraction between c (K, T) and any loop in codes (N, K, T), and its index."""
    match = (codes == c.unsqueeze(0)).float().mean(dim=(1, 2))        # fraction of identical tokens
    best = match.max(dim=0)
    return best.values.item(), best.indices.item()


# Reference: how similar is a real loop to its closest *other* real loop? (first 100 loops)
ref = []
for i in range(100):
    m = (codes == codes[i].unsqueeze(0)).float().mean(dim=(1, 2))
    m[i] = 0                                          # exclude the loop itself
    ref.append(m.max().item())
print("typical real-to-real nearest match:", sum(ref) / len(ref))

# Compare with the last sample generated above (final setting, last seed).
print("generated sample:", nearest_loop(last_sample, codes))