import torch
import torchaudio
from torchaudio import transforms
import numpy as np

class AudioUtil:
    @staticmethod
    def open(audio_file):
        """ Load an audio file. Return the signal as a tensor and the sample rate """
        sig, sr = torchaudio.load(audio_file)
        return sig, sr


    @staticmethod
    def pad_trunc(audio, max_seconds):
        """ Pad (or truncate) given signal to a fixed length 'max_seconds' in seconds """
        signal, sampling_rate = audio
        num_rows, sig_len = signal.shape
        max_len = sampling_rate * max_seconds
        if sig_len > max_len:
            signal = signal[:, :max_len]
        elif sig_len < max_len:
            pad_begin_len = np.random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            signal = torch.cat((pad_begin, signal, pad_end), 1)
        return signal, sampling_rate



    @staticmethod
    def generate_spectrogram(aud):
        """ Generate a Mel Spectrogram given a pair (signal, sample rate)"""
        sig, sr = aud

        # Using Mel Spectogram
        # top_db = 80
        # spec = transforms.MelSpectrogram(sr,  n_mels=64, n_fft=1024, hop_length=None)(sig.float())
        # mel_spec = transforms.AmplitudeToDB(top_db=top_db).forward(spec)

        # Using MFCC
        mfcc = transforms.MFCC(sample_rate=sr)(sig.float())
        return mfcc


    @staticmethod
    def add_noise(audio):
        """ Add uniform noise to audio """
        signal, sampling_rate = audio
        _, sig_len = signal.shape
        signal = signal + np.random.uniform(-0.05, 0.05, sig_len)
        return signal, sampling_rate


    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    # @staticmethod
    # def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    #     _, n_mels, n_steps = spec.shape
    #     mask_value = spec.mean()
    #     aug_spec = spec
    #
    #     freq_mask_param = max_mask_pct * n_mels
    #     for _ in range(n_freq_masks):
    #         aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    #
    #     time_mask_param = max_mask_pct * n_steps
    #     for _ in range(n_time_masks):
    #         aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
    #
    #     return aug_spec
