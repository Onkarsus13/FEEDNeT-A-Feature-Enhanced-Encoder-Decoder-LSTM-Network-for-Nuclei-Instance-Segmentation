from model import *
from dataset import *
from config import config as cfg

#Train function to traing and save the model
def train(epochs):
    train_ds, test_ds = get_data()
    model = Exsumsion_net(pretrained_weights=False)
    model.summary()

    for i in range(epochs):
        print('{}/500'.format(i))
        his = model.fit(train_ds, epochs=1, verbose = 1, validation_data=test_ds)
        if i%5==0:
            model.save('LSTM_UNET.h5')

if __name__ == "__main__":
    train(cfg.epochs)

