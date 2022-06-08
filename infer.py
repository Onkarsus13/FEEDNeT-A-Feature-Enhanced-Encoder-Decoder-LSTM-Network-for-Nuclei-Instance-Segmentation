from model import *
from config import config as cfg
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

#this is the inference function for to test our model
def inference(image_path):
    model = Exsumsion_net(pretrained_weights=True)
    image = cv2.imread(image_path)
    print(image.shape)
    image = np.expand_dims(image, axis=0)
    image = image/255.0
    pred = model.predict(image)
    pred = 1 - (pred[0,:,:,0]>0.5).astype('uint8')
    plt.imsave('test.png', pred, cmap='gray')


if __name__ == '__main__':
    inference('data/Train/images/image_7.png')


