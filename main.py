from data_loader import Loader
from model import AudioClassifier
from training import train
import torch

PATH = '/home/maksim/Data/voice_clf/'

def main():
    loader = Loader(PATH)
    train_dl, dev_dl, test_dl = loader.load_dataset()
    mymodel = AudioClassifier()
    train(mymodel, train_dl, num_epochs=5)
    torch.save(mymodel.state_dict(), PATH + "augment_spectrogram.pth")
    return 0

if __name__ == '__main__':
    main()
