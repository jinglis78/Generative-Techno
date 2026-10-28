# Generative techno drum loops

Experiments in generating two-bar, 140 BPM techno drum loops with three different models:

| Approach | Script | Representation | Model |
|----------|--------|----------------|-------|
| Convolutional VAE | `main.py` | dB magnitude spectrogram | `StridedConvVAE` |
| Frame transformer | `transformer.py` | dB magnitude spectrogram, one frame = one vector | `FrameTransformer` (causal) |
| Codec-token transformer | `transformer_codec.py` | EnCodec tokens (3 codebooks) | `TokenTransformer` (causal) |

The two spectrogram models are inverted to audio with Griffin-Lim. The token model is decoded with EnCodec.

This is research / experiment code, not a packaged library. Each script is its own entry point.

## Repository layout

| File | Role |
|------|------|
| `main.py` | VAE: dataset selection, spectrogram conversion, training, logging, sampling |
| `vae_model.py` | `StridedConvVAE` (used) and an earlier `GenerativeVAE` (fixed layer sizes; adds per-bin noise from `sigma_vae3.pt` at decode) |
| `vae_base_class.py` | VAE training loop and loss (L1 reconstruction + frequency-difference term + KL) |
| `data_processing_class.py` | Load audio, STFT, normalise / denormalise, Griffin-Lim, plot and write files |
| `transformer.py` | Frame-level causal transformer on spectrograms: training, teacher-forced reconstruction, autoregressive generation |
| `transformer_codec.py` | Causal transformer on EnCodec tokens: delay pattern, beat-phase features, training, sampling, memorisation check |
| `dpte.py` | Standalone helper that counts two-bar loops in the sample packs |

Generated artefacts (`*.pth`, `output.txt`, spectrogram caches, plots, audio) are gitignored.

## Requirements

- Python 3.10+ recommended
- A machine with enough RAM to hold the spectrogram tensor; Apple Silicon uses MPS if available, otherwise CPU

Install:

```
pip install torch torchaudio torchvision librosa soundfile matplotlib scipy numpy ipython encodec
```

`encodec` is only needed for `transformer_codec.py`; the pretrained 24 kHz model is downloaded on first use. On macOS, install the official PyTorch wheels from [pytorch.org](https://pytorch.org) if you want MPS acceleration.

## Data

The sample packs are **not** in this repo. `main.py` currently points at local paths:

- Dave Parkinson Trance Essentials Vol. 2 drum loops
- Dave Parkinson Trance Essentials 140 BPM drum loops
- A converted 140 BPM folder under `generative_audio/`

Folders named `DPTE2 Snare Builds` and `DPTE2 Percussion FX Loops` are skipped. Only files whose duration rounds to `3.43` seconds (two bars at 140 BPM) are kept. Edit `audio_dirs` (and the converted-folder path) in `main.py` to match your machine. `transformer_codec.py` does not read the audio packs; it works from pre-computed tokens (see section 3).

A commented-out block in `main.py` can time-stretch 134-139 BPM loops to 140 BPM (pitch preserved) to produce the converted folder.

### Spectrogram settings

- 22,500 Hz sample rate, window of 1024 samples (about 45.5 ms), hop of 261 samples (about 11.6 ms)
- Each loop becomes a `(513 bins, 296 frames)` magnitude spectrogram in dB
- Values are scaled into `[0, 1]` with `(dB + 100) / 100`

`main.py` writes the cache `specgrams.pkl` to `output_dir` (see below); later runs reuse it if it still exists. The dataset used so far has 1,756 loops.

## 1. Convolutional VAE (`main.py`)

Trains `StridedConvVAE`: a 2D convolutional encoder, a latent vector, and a transposed-convolution decoder, on the `(1, 513, 296)` spectrograms.

```
python main.py
```

`main.py` redirects stdout/stderr to `output.txt` in the same folder.

Current settings in `main.py`:

- batch size 16, 20 epochs, Adam `lr=0.001`
- latent dim 64
- 6 conv layers, kernels `(5, 5, 5, 5, 5, 5)`
- strides `((2,2), (2,2), (2,2), (1,1), (1,1), (1,1))`
- channels `(16, 32, 64, 128, 256, 256)`

Loss (`vae_base_class.py`): `L1 reconstruction + 100 * frequency-difference L1 + beta * KL`, with `beta = 0.001`. The frequency-difference term compares differences between neighbouring frequency bins of the reconstruction and the input.

**Checkpoint:** `main.py` loads `vae_model_DPTE_vae3.pth` before training, so it resumes from that file and fails if it does not exist. To train from scratch, comment out the `torch.load(...)` / `load_state_dict(...)` lines. A checkpoint must match the architecture settings above. After training it saves weights back to `vae_model_DPTE_vae3.pth`.

Outputs, written to `output_dir` (edit it in `main.py`):

```
.../generative_audio/DPTE
```

- `losses.png`: training curve
- generated samples via `data.plot_samples(...)`
- `experiment_log.txt` (in the working directory): hyperparameters, loss function source, and per-epoch reconstruction / frequency / KL terms

Note: the VAE is evaluated on the same loader it trains on (no held-out split).

## 2. Frame transformer (`transformer.py`)

A decoder-only (causal) transformer that treats each spectrogram frame as one token-vector.

**Input.** Reads the cached spectrograms from `./DPTE/specgrams.pkl` (a path relative to where you run the script), converts them with `(spec + 100) / 100`, and transposes to `(N, 296, 513)`: 296 frames, each a 513-dim vector.

**Model (`FrameTransformer`).**
- Linear layer maps each 513-bin frame to a 256-d vector
- Learned positional embedding, one vector per frame index (296)
- 4 pre-norm transformer encoder layers, 8 heads, feed-forward width 1024, with a causal attention mask
- Linear layer + sigmoid maps back to 513 bins; the output at position `t` is the prediction of frame `t+1`

**Training.** L1 loss on next-frame prediction. 100 loops held out for validation (seed 0), batch 16, 20 epochs, AdamW (`lr=3e-4`, weight decay 0.01), cosine schedule, gradient clipping at 1.0. The script also prints the L1 of a copy-the-previous-frame baseline.

**Running.** Set `TRAIN` at the top of the script. As committed `TRAIN = False`, so it loads `frame_transformer.pth` and goes straight to evaluation and generation:

```
python transformer.py
```

Set `TRAIN = True` to train from scratch; the weights are then saved to `frame_transformer.pth`.

**Outputs.**
- Teacher-forced reconstruction of a validation loop: `teacher_forced.png` (original / reconstruction / difference), `orig.wav`, `recon.wav`, and `recon-orig.wav` (the difference, amplified)
- Autoregressive generation: the first 37 frames (about one beat) of a held-out loop seed the model, which then generates the remaining frames to 296 by feeding each predicted frame back in. The model is deterministic, so there is no sampling. Writes `autoregressive.png`, `orig_gen.wav`, `recon_gen.wav`, and `recon-gen.wav`

Spectrograms are converted to audio with `torchaudio` Griffin-Lim (200 iterations, momentum 0.99, random init) after undoing the dB scaling, with a x2 output gain.

## 3. EnCodec token transformer (`transformer_codec.py`)

A decoder-only transformer over discrete EnCodec tokens, with a beat-phase signal for 140 BPM.

**Tokenisation.** EnCodec 24 kHz at a 6 kbps target bandwidth (8 codebooks, 75 frames/s). Only the first `K = 3` codebooks are used. Vocabulary is 1025: 1024 codes plus a padding id (`1024`). A 3.43 s loop is about 257 token frames.

**Prerequisite.** The script loads pre-computed tokens from `codec_tokens.pt` (one row per two-bar loop, shape `(N, codebooks, T)`), which it slices to the first `K` codebooks. The file was extracted once with earlier code and cached so the loops don't need re-encoding; keep it in the working directory (or regenerate it by encoding each loop with EnCodec).

**Delay pattern.** Codebook `k` is shifted `k` steps later (padded with the pad id), so one step predicts the first codebook of the current frame and finer codebooks of earlier frames. Sequences become length `T + K - 1`. The inputs are the targets shifted right by one step.

**Model (`TokenTransformer`).**
- One embedding table per codebook (embeddings are summed), model width 384
- Learned positional embedding (max length 300)
- Beat-phase features: sin and cos of the position in beats at periods of 1/4, 1/2, 1, 2, 4 and 8 beats (12 features, based on about 32.14 frames per beat), projected linearly to the model width
- 6 pre-norm transformer layers, 8 heads, feed-forward width 1536, dropout 0.1, causal mask
- One output head per codebook (`Linear(384, 1025)`)

**Training.** Cross-entropy summed over codebooks, ignoring pad targets. 100 loops held out (seed 0), batch 16, 15 epochs, AdamW (`lr=3e-4`, weight decay 0.01), cosine schedule, gradient clipping at 1.0. Prints train / validation loss each epoch and saves `token_transformer_cd3_beat.pth`. Set `TRAIN = False` at the top of the script to load that checkpoint instead of training.

```
python transformer_codec.py
```

**Generation.** Tokens are sampled autoregressively with temperature and top-k (defaults 0.8 and 100), the delay pattern is undone, and EnCodec decodes the result. The script writes:

- `gen_1cb.wav`, `gen_2cb.wav`: one sample decoded with only the first 1 or 2 codebooks
- `gen_t{temperature}_k{top_k}_s{seed}.wav`: 3 seeded samples for each of four (temperature, top-k) settings: (0.6, 100), (1.0, 250), (0.9, 100), (0.8, 50)

**Memorisation check.** For a generated sample, it reports the fraction of tokens that match its nearest training loop, next to a reference value: how closely a real loop matches its nearest *other* real loop.

## Notes

- All data and output paths are hard-coded to a local machine; edit them before running.
- Run scripts from this directory.
- `dpte.py` is optional: it prints how many two-bar loops each pack contains, without training.

```
python dpte.py
```