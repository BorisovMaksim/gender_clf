import numpy as np
import pandas as pd
from pathlib import Path
import os
from dataset_generator import SoundDS
import torch.utils.data


class Loader:
    def __init__(self, path):
        self.path = path
        self.train_path = os.path.join(path, 'train-clean-100/LibriTTS/train-clean-100/')
        self.dev_path = os.path.join(path, 'dev-clean/LibriTTS/dev-clean/')
        self.test_path = os.path.join(path, 'test-clean/LibriTTS/test-clean/')

    def load_speakers_data(self):
        """ Read data about speakers and return dictionary [reader_id -> gender]"""
        data_speakers = pd.read_table(os.path.join(self.path, "train-clean-100/LibriTTS/speakers.tsv"), sep='\t')
        speakers = data_speakers.reset_index()
        speakers.columns = ["READER", "GENDER", "SUBSET", "NAME"]
        speakers['is_M'] = (speakers["GENDER"] == "M") + 0.0
        dict_reader_to_gender = pd.Series(speakers.is_M.values, index=speakers.READER).to_dict()
        return dict_reader_to_gender

    def load_paths(self):
        """ Parse filenames from data directory in random order"""
        rng = np.random.default_rng(seed=42)

        train_files = np.array(list(Path(self.train_path).rglob('*/*/*.wav')))
        dev_files = np.array(list(Path(self.dev_path).rglob('*/*/*.wav')))
        test_files = np.array(list(Path(self.test_path).rglob('*/*/*.wav')))

        rng.shuffle(train_files)
        rng.shuffle(dev_files)
        rng.shuffle(test_files)

        return train_files, dev_files, test_files

    def load_dataset(self):
        """ Loading all the necessary data and output Data Loaders for train/dev/test"""

        dict_reader_to_gender = self.load_speakers_data()
        train_files, dev_files, test_files = self.load_paths()

        # reader_id of audiofile = int(file.parts[-3])
        y_train = np.array([dict_reader_to_gender[int(file.parts[-3])] for file in train_files])
        y_dev = np.array([dict_reader_to_gender[int(file.parts[-3])] for file in dev_files])
        y_test = np.array([dict_reader_to_gender[int(file.parts[-3])] for file in test_files])

        train_ds = SoundDS(train_files, y_train)
        dev_ds = SoundDS(dev_files, y_dev)
        test_ds = SoundDS(test_files, y_test)

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
        dev_dl = torch.utils.data.DataLoader(dev_ds, batch_size=16, shuffle=False)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

        return train_dl, dev_dl, test_dl
