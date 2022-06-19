from model import *
from dataset import *
from config import config as cfg

#Train function to traing and save the model
def train(epochs):
    train_ds, test_ds = get_data()
    model = FEEDNet(pretrained_weights=False, seg_type='binary')
    model.summary()

    for i in range(epochs):
        print('{} epoch'.format(i))
        his = model.fit(train_ds, epochs=1, verbose = 1, validation_data=test_ds)
        if i%5==0:
            model.save('File_Name.h5') #put the name of file by which you want to save the trained models weights

if __name__ == "__main__":
    train(cfg.epochs)

