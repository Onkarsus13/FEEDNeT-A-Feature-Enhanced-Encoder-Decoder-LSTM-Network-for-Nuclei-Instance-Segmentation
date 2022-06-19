import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import *
from config import config as cfg


def downsample_image(x,shape):
    return tf.image.resize(x,[shape,shape])

def diceCoef(y_true, y_pred):   
    smooth = 1.
    y_true_f = K.flatten(y_true)    
    y_pred_f = K.flatten(y_pred)    
    intersection = K.sum(y_true_f * y_pred_f)    
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def diceCoefLoss(y_true, y_pred):
    return 1.-diceCoef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + diceCoefLoss(y_true, y_pred) #+ tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    return loss

def FEEDNet(pretrained_weights = None, input_size = (256,256,3), seg_type=cfg.seg_type):
    inputs = Input(shape = input_size)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    down1 = downsample_image(inputs, 128)
    conv_d1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down1) 
    concat1 = concatenate([conv_d1, pool1]) 

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat1)
    conv2 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    down2 = downsample_image(inputs, 64)
    conv_d2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down2)
    concat2 = concatenate([conv_d2, pool2])

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat2)
    conv3 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    down3 = downsample_image(inputs, 32)
    conv_d3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down3)
    concat3 = concatenate([conv_d3, pool3])

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat3)
    conv4 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.4)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    down4 = downsample_image(inputs, 16)
    conv_d4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down4)
    concat4 = concatenate([conv_d4, pool4])

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat4)
    conv5 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.4)(conv5)

    re = Reshape((256, 512))(drop5)
    lstm = LSTM(512, return_sequences=True)(re)
    re2 = Reshape((16, 16, 512))(lstm)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(re2))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    if seg_type == 'multiclass':
        conv10 = Conv2D(8, 1, activation = 'sigmoid', name = 'segmentation')(conv9)
    elif seg_type == 'binary':
        conv10 = Conv2D(2, 1, activation = 'sigmoid', name = 'segmentation')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(learning_rate = cfg.lr), 
                  loss = {'segmentation':bce_dice_loss},
                  metrics = {'segmentation':['accuracy', diceCoef]},
                  run_eagerly=True
                  )

    if(pretrained_weights):
    	model.load_weights('data/FEEDNet.h5') #put the path of corresponding weight file here

    return model