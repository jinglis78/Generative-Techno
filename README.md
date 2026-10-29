# Generative techno drum loops

Experiments in generating two-bar 140 BPM techno drum loops with three different models:

| Model | Script | Descripton |
|----------|--------|-----------------------|
| Convolutional VAE | `main.py` | Uses a strided convolutional VAE to model |
| Frame transformer | `transformer.py` | Give a history of frames, predicts the next frame. | `FrameTransformer` |
| Codec-token transformer | `transformer_codec.py` | EnCodec tokens (3 codebooks) | `TokenTransformer` |

The models are trained on approximately a local dataset of approximately 2000 two-bar 140 BPM techno drum loops. 
This dataset is **not** in this repo, the samples are obtained from the Dave Parkinson Trance Essentials sample packs.
The raw `.wav` files are converted to dB magnitude spectrograms, and each model is trained to reproduce the spectrograms.

The convolutional VAE and frame transformer models suffer from ``metallic-sounding'' reconstructed and generated audio. 
This was not an artifact of the spectrogram-to-waveform conversion via Griffin-Lim, as the reconstructed/generated spectrograms show a smoother, less-noisy structure compared to the orginal spectrogram. 
The original spectrograms do not have the same problem when reconstructed using Griffin-Lim.


It was seen that the [EnCodec model](https://audiocraft.metademolab.com/encodec.html) is much more effective at reconstructing audio. This model proceeds as follows:

The encoder first converts audio into a sequence of 128-dimensional latent vectors - there is a single vector for every 13.33 ms.
Each vectors is converted into 8 discrete integer tokens via 8 codebooks, which contain 1024 learned vectors.
The model selects the vector in the frist codebook that most closely approximates the original latent vector. 
The additional codebooks selects the vector that most closely resemble subsequent residuals between the original and codebook latent vectors.
The decoder uses the 8 selected vectors to form an approximation of the original latent representation, then converts that representation back into a waveform.

The codec-token transformer model uses the encodor and the first `K=3` codebooks from EnCodec to convert each techno loop to a series of codec tokens. The transformer model is trained to reproduce these token sequences. The pretrained decoder turns those tokens into audio. A beat-phase signal is included in the positional encoding, so the model has a direct clock for where it is in the beat instead of learning it.

## Repository layout

| File | Role |
|------|------|
| `main.py` | VAE: dataset selection, spectrogram conversion, training, logging, sampling |
| `vae_model.py` | `StridedConvVAE` (used) and an earlier `GenerativeVAE` (fixed layer sizes; adds per-bin noise from `sigma_vae3.pt` at decode) |
| `vae_base_class.py` | VAE training loop and loss (L1 reconstruction + frequency-difference term + KL) |
| `data_processing_class.py` | Load audio, STFT, normalise / denormalise, Griffin-Lim, plot and write files |
| `transformer.py` | Frame-level transformer on spectrograms: training, teacher-forced reconstruction, autoregressive generation |
| `transformer_codec.py` | Transformer on EnCodec tokens: delay pattern, beat-phase features, training, sampling, memorisation check |
