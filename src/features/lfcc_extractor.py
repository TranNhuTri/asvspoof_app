import torchaudio
from torchaudio import transforms

from src.configs import lfcc_config


def extract_lfcc_features(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
    win_length = int((sample_rate / 1000) * lfcc_config.WIN_LENGTH)
    transform = transforms.LFCC(
        sample_rate=sample_rate,
        n_filter=lfcc_config.N_LFCC,
        n_lfcc=lfcc_config.N_LFCC,
        speckwargs={"n_fft": lfcc_config.N_FFT, "win_length": win_length},
    )
    features = transform(waveform)[0].numpy()
    return features
