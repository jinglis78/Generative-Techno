import torchaudio, torch
import torchaudio.functional as taF
from pathlib import Path
import matplotlib.pyplot as plt
import IPython.display as ipd
from math import ceil,floor,sqrt
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import pickle
import shutil
from IPython.display import display

from pathlib import Path
from scipy.io.wavfile import write
import numpy as np

class DataProcessor:
    def __init__(self, audio_paths, output_dir,  samplerate, window_duration, hop_duration, audio_duration, device):
        # data_path = list of Path objects, each representing a directory path for a .wav file in the dataset
        self.audio_paths = audio_paths
        self.num_files = len(audio_paths)

        # where the spectrogram plots and reconstructed .wav files will be stored
        self.specgrams_path = Path(output_dir+'/specgrams.pkl')
        self.plot_dir = Path(output_dir+'/plots/')
        self.audio_dir = Path(output_dir+'/audio/')
        if self.audio_dir.exists():
            shutil.rmtree(self.audio_dir)

        if self.plot_dir.exists():
            shutil.rmtree(self.plot_dir)


        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # audio specific parameters
        self.samplerate = samplerate
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        self.audio_duration = audio_duration

        self.audio_size = self.audio_duration * self.samplerate
        self.window_size = round(self.window_duration*self.samplerate)
        self.hop_size = round(self.hop_duration*self.samplerate)
        self.num_bins = self.window_size // 2 + 1
        self.num_frames = ceil(self.audio_duration / self.hop_duration)

        self.spec_max = None
        self.spec_min = None

        self.device = device

    def plot_samples(self, model, num_samples, latent_dim):
        model.eval()
        with torch.no_grad():
            try:
                randn_shape = (num_samples,) + tuple(latent_dim)
            except TypeError:
                randn_shape = (num_samples,) + (latent_dim,)
            random_input = torch.randn(randn_shape).to(self.device)
            samples = model.decode(random_input)

        
        for i,normed_specgram in enumerate(samples[:5]):   
            waveform = self.reconstruct_waveform(normed_specgram.cpu(), self.window_size, self.hop_size)
        
            # Convert tensors or arrays to NumPy
            waveform = np.asarray(waveform).squeeze()

            # Save as 16-bit WAV files
            write(
                self.audio_dir / f"generated_{i+1}.wav",
                self.samplerate,
                (np.clip(waveform, -1, 1) * 32767).astype(np.int16)
            )


    def extract_spectrogram(self, audio_path, window_size, hop_size, samplerate, n_fft=None):
        samples, original_rate = torchaudio.load(audio_path)
        mono_samples = samples.mean(dim=0, keepdim=False)
        resampled_audio = taF.resample(mono_samples, original_rate, samplerate)
        if n_fft is None:
            n_fft = window_size
        specgram = torch.stft(
            resampled_audio, 
            n_fft=window_size, 
            window=torch.hann_window(window_size), 
            hop_length=hop_size, 
            normalized=False, 
            return_complex=True).abs().squeeze()
        dB_specgram = 20 * torch.log10(specgram.clamp(min=1e-5))
        #dB_specgram = taF.amplitude_to_DB(specgram, multiplier=20, amin=1e-5, db_multiplier=specgram.max().log10())
        return dB_specgram
    
    # convert the dataset to spectrograms
    def convert_to_spectrograms(self):
        print (self.specgrams_path, self.specgrams_path.exists())
        spec_data = None
        if self.specgrams_path.exists():
            with open(self.specgrams_path, 'rb') as f:
                spec_data = pickle.load(f)
        else:
            spec_data = torch.full((self.num_files, self.num_bins, self.num_frames), -100) # for storing spectrograms
            for i, audio_path in enumerate(self.audio_paths):
                if i % 100 == 0:
                    print(f'Processing file {i+1}/{self.num_files}')
                specgram = self.extract_spectrogram(audio_path, self.window_size, self.hop_size, self.samplerate)
                spec_data[i, :, :specgram.shape[1]] = specgram[:, :self.num_frames] # truncate if necessary
            with open(self.specgrams_path, 'wb') as f:
                pickle.dump(spec_data, f)

        self.spec_min = spec_data.min()
        self.spec_max = spec_data.max()

        return spec_data

    def plot_spectrogram(self, specgram, output_file):
        specgram = specgram.detach().cpu().squeeze().numpy()
        fig, ax = plt.subplots()
        img = ax.imshow(specgram, aspect='auto', origin='lower')
        fig.colorbar(img)
        ax.set_xlabel('frames')
        ax.set_ylabel('frequency bins')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # normalise the spectrogram such that all values fall within [0,1]
    def normalise_specgram(self, specgram):
        spec_min = -100
        spec_max = 0
        normed_specgram = (specgram - spec_min) / (spec_max - spec_min)
        return normed_specgram

    # denormalise the spectrogram 
    def denormalise_specgram(self, normed_specgram):
        min_value = -100
        max_value = 0
        denormed_specgram = normed_specgram.squeeze() * (max_value - min_value) + min_value
        return denormed_specgram
    
    def reconstruct_waveform(self, normed_specgram, window_size, hop_size):
        denormed_specgram = self.denormalise_specgram(normed_specgram)

        # Invert 20*log10(magnitude)
        magnitude = 10.0 ** (denormed_specgram / 20.0)

        window = torch.hann_window(
            window_size,
            device=magnitude.device,
            dtype=magnitude.dtype
        )

        waveform = taF.griffinlim(
            magnitude,
            window=window,
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            power=1.0,
            n_iter=64,
            momentum=0.99,
            length=None,
            rand_init=True
        )

        waveform = waveform * 2.0

        return waveform

    def display_reconstruction(self, data, recon_batch, epoch):
        n = min(len(data), 2)
        for i in range(n-1):
            original_waveform = self.reconstruct_waveform(data[i].cpu(), self.window_size, self.hop_size)
            reconstructed_waveform = self.reconstruct_waveform(recon_batch[i].cpu(), self.window_size, self.hop_size)

            # Convert tensors or arrays to NumPy
            original = np.asarray(original_waveform).squeeze()
            reconstructed = np.asarray(reconstructed_waveform).squeeze()

            self.plot_spectrogram(data[i].cpu(), self.plot_dir / f"original_{i+1}_epoch{epoch}.png", )
            self.plot_spectrogram(recon_batch[i].cpu(), self.plot_dir / f"recon_{i+1}_epoch{epoch}.png", )
            
            # Save as 16-bit WAV files
            write(
                self.audio_dir / f"original_{i+1}_epoch{epoch}.wav",
                self.samplerate,
                (np.clip(original, -1, 1) * 32767).astype(np.int16)
            )

            write(
                self.audio_dir / f"reconstructed_{i+1}_epoch{epoch}.wav",
                self.samplerate,
                (np.clip(reconstructed, -1, 1) * 32767).astype(np.int16)
            )