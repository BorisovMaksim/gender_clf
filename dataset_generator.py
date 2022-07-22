from torch.utils.data import Dataset
from audio_utils import AudioUtil



class SoundDS(Dataset):
    """ Map-style dataset object, containing pairs (Spectrogram, label) """
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
        self.duration_seconds = 6

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Read audiofile from disk and preprocess it using:
            1. Pad or truncate  to length of 6 seconds
            2. Add noise
            3. Calculate Spectrogram
        """
        audio_file = self.paths[idx]
        label = self.labels[idx]

        aud = AudioUtil.open(audio_file)
        dur_aud = AudioUtil.pad_trunc(aud, self.duration_seconds)
        noised_aud = AudioUtil.add_noise(dur_aud)
        sgram = AudioUtil.generate_spectrogram(noised_aud)
        # Maybe will be interesting to use spectrogram augmentation in the future work
        # aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return sgram, label
