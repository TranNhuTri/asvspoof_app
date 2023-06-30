import librosa
import numpy as np

from src.features.lfcc_extractor import extract_lfcc_features


def extract_delta_spectral_cepstral_features(audio_path):
    lfcc = extract_lfcc_features(audio_path)
    lfcc_delta = librosa.feature.delta(lfcc)
    lfcc_delta2 = librosa.feature.delta(lfcc, order=2)
    features = np.concatenate((lfcc, lfcc_delta, lfcc_delta2), axis=1)
    return features
