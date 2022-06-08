from utils import *
import glob
from tensorflow.keras.utils import Sequence
import cv2
import albumentations as A
from config import config as cfg


def make_list(s):
  l = sorted(s)
  return l

imagesTr  = glob.glob(cfg.train_images)
imagesTs  = glob.glob(cfg.test_images)
labelsTr  = glob.glob(cfg.train_mask)
labelsTs = glob.glob(cfg.test_masks)


mages_train = make_list(imagesTr)
masks_train = make_list(labelsTr)
images_test = make_list(imagesTs)
masks_test = make_list(labelsTs)


#Data Generator Class
class CoNSePDataset(Sequence):
  def __init__(self, img_paths=None, mask_paths=None, a = True):
    self.img_paths = img_paths
    self.mask_paths = mask_paths
    self.a = a
    self.transform = A.Compose([      
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.4),
    A.Transpose(p=0.5),          
    A.RandomRotate90(p=0.5),
    A.GridDistortion(p=0.3),
    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.RandomBrightnessContrast(p=0.4),    
    A.RandomGamma(p=0.5)
    ])
    assert len(self.img_paths) == len(self.mask_paths)
    self.images = len(self.img_paths) #list all the files present in that folder...
  
  def __len__(self):
    return len(self.img_paths) #length of dataset
  
  def __getitem__(self, index):
    img_path = self.img_paths[index]
    mask_path = self.mask_paths[index]
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if self.a:
        augment = self.transform(image = image, mask=mask)
        image = augment['image']
        mask = augment['mask']

    image = image.astype(np.float32)
    image = image/255.0
    
    mask = mask.astype(np.float32)
    mask = rgb_to_onehot(mask)
    mask = np.expand_dims(mask, axis = 0)
    return np.expand_dims(image, axis = 0), mask

#this function we call during traing to get set of dataloader i.e train loader and val_loader
def get_data():
    train_ds = CoNSePDataset(
            img_paths=imagesTr,
            mask_paths=labelsTr,
            a = True
        )

    test_ds = CoNSePDataset(
            img_paths=imagesTs,
            mask_paths=labelsTs,
            a = False)
    
    return train_ds, test_ds