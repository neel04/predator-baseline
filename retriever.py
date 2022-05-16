import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import cv2
import albumentations as A
from albumentations.core.composition import Compose

from typing import Callable, List
from pathlib import Path
from torch.utils.data import Dataset
import torch
import sys

def pad_to_multiple(x, k=32):
    return int(k*(np.ceil(x/k)))

def get_train_transforms(height: int = 437, 
                         width: int = 582, 
                         level: str = 'hard'): 
    if level == 'light':
        return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.IAAAdditiveGaussianNoise(p=0.2),
                A.OneOf(
                    [A.CLAHE(p=1.0),
                    A.RandomBrightness(p=1.0),
                    A.RandomGamma(p=1.0),
                    ],p=0.5),
                A.OneOf(
                    [A.IAASharpen(p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                    ],p=0.5),
                A.OneOf(
                    [A.RandomContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    ],p=0.5),
                A.Resize(height=height, width=width, p=1.0),
                A.PadIfNeeded(pad_to_multiple(height), 
                              pad_to_multiple(width), 
                              border_mode=cv2.BORDER_CONSTANT, 
                              value=0, 
                              mask_value=0)
            ], p=1.0)

    elif level == 'hard':
        return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.IAAAdditiveGaussianNoise(p=0.2),
                A.OneOf(
                    [A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                     A.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                     A.ShiftScaleRotate(
                         shift_limit=0,
                         scale_limit=0,
                         rotate_limit=10,
                         border_mode=cv2.BORDER_CONSTANT,
                         value=0,
                         mask_value=0,
                         p=1.0
                     ),
                     A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    ],p=0.5),
                A.OneOf(
                    [A.CLAHE(p=1.0),
                    A.RandomBrightness(p=1.0),
                    A.RandomGamma(p=1.0),
                    A.ISONoise(p=1.0)
                    ],p=0.5),
                A.OneOf(
                    [A.IAASharpen(p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                    ],p=0.5),
                A.OneOf(
                    [A.RandomContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    ],p=0.5),
                A.Resize(height=height, width=width, p=1.0),
                A.Cutout(p=0.3),
                A.PadIfNeeded(pad_to_multiple(height), 
                              pad_to_multiple(width), 
                              border_mode=cv2.BORDER_CONSTANT, 
                              value=0, 
                              mask_value=0) 
            ], p=1.0)
    elif level == 'hard_weather':
        return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.IAAAdditiveGaussianNoise(p=0.2),
                A.OneOf(
                    [A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                     A.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                     A.ShiftScaleRotate(
                         shift_limit=0,
                         scale_limit=0,
                         rotate_limit=10,
                         border_mode=cv2.BORDER_CONSTANT,
                         value=0,
                         mask_value=0,
                         p=1.0
                     ),
                     A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    ],p=0.5),
                A.OneOf(
                    [A.CLAHE(p=1.0),
                    A.RandomBrightness(p=1.0),
                    A.RandomGamma(p=1.0),
                    A.ISONoise(p=1.0)
                    ],p=0.5),
                A.OneOf(
                    [A.IAASharpen(p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                    ],p=0.5),
                A.OneOf(
                    [A.RandomContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    ],p=0.5),
                A.OneOf(
                    [A.RandomFog(fog_coef_upper=0.8, p=1.0),
                     A.RandomRain(p=1.0),
                     A.RandomSnow(p=1.0),
                     A.RandomSunFlare(src_radius=100, p=1.0)
                    ],p=0.4),
                A.Resize(height=height, width=width, p=1.0),
                A.Cutout(p=0.3),
                A.PadIfNeeded(pad_to_multiple(height), 
                              pad_to_multiple(width), 
                              border_mode=cv2.BORDER_CONSTANT, 
                              value=0, 
                              mask_value=0) 
            ], p=1.0)

def get_valid_transforms(height: int = 437, 
                         width: int = 582): 
    return A.Compose([
            A.Resize(height=height, width=width, p=1.0),
            A.PadIfNeeded(pad_to_multiple(height), 
                          pad_to_multiple(width), 
                          border_mode=cv2.BORDER_CONSTANT, 
                          value=0, 
                          mask_value=0)
        ], p=1.0)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn: Callable):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

#Precomputing superpixels
#Run this once, unless already computed by providing all the paths to images
def superpixels_precom(paths):
    print(f'\nPrecomputing superpixels... might take hours :(')
    exec_bash(f'mkdir ./comma10k/superpixels')

    for img_path in tqdm(paths):
        og_image = cv2.imread(img_path)
        src_image = cv2.resize(og_image, (256, 256))  

        segments = slic(src_image, n_segments=1500, sigma=1, compactness=2, multichannel=True)
        superpixels = color.label2rgb(segments, src_image, kind='avg')

        cv2.imwrite(f'./comma10k/superpixels/{img_path.split("/")[-1]}', superpixels)

def algo_preprocessor(image, img_path):
    '''
    returns a preprocessed image based on SLIC (Superpixels) and Canny algorithm
    '''
    #Cannying the image
    src_image = cv2.resize(image, (256, 256)) #ensure the image is 256x256
    
    #convert image to gray for cannying
    gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    img_blur = np.uint8(cv2.GaussianBlur(gray_image, (3,3), 0))
    canny_edges = cv2.cvtColor(cv2.Canny(image=img_blur, threshold1=30, threshold2=50), cv2.COLOR_BGR2RGB) # Canny Edge Detection

    #loading precomputed superpixels
    #check if path exists for switching between Kaggle and Colab
    base_path = '../input/comma10k/'

    if not os.path.exists(base_path):
        base_path = './comma10k/'
    superpixels = cv2.cvtColor(cv2.imread(f'{base_path}superpixels/{img_path.split("/")[-1]}'), cv2.COLOR_BGR2RGB)
    
    #convert hwc to chw for concatenation
    #superpixels = np.transpose(superpixels, (2, 0, 1))
    #canny_edges = np.transpose(canny_edges, (2, 0, 1))
    return superpixels, canny_edges

class TrainRetriever(Dataset):

    def __init__(self, 
                 data_path: Path, 
                 image_names: List[str], 
                 preprocess_fn: Callable, 
                 transforms: Compose,
                 class_values: List[int]):
        super().__init__()
        
        self.data_path = data_path
        self.image_names = image_names
        self.transforms = transforms
        self.preprocess = get_preprocessing(preprocess_fn)
        self.class_values = class_values
        self.images_folder = 'imgs'
        self.masks_folder = 'masks'

    def __getitem__(self, index: int):
        
        image_name = self.image_names[index]
        
        image = cv2.imread(str(self.data_path/self.images_folder/image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #getting preprocessed images
        superpixel_image, cannied_image = algo_preprocessor(image, image_name) #image_name is supposed to be the path
        mask = cv2.imread(str(self.data_path/self.masks_folder/image_name), 0).astype('uint8')

        if self.transforms:
            sample = self.transforms(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']

        mask = np.stack([(mask == v) for v in self.class_values], axis=-1).astype('uint8')

        if self.preprocess:
            sample = self.preprocess(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
            cannied_image = self.preprocess(image=cannied_image)['image'] #extracting the preprocessed frame
        
        image = image.astype('float32')
        cannied_image = cannied_image.astype('float32')

        final_input_image = np.concatenate([image, cannied_image], axis=0)
        #final_input_image = cv2.addWeighted(image, 0.8, cannied_image, 0.20, 0.5).transpose(2,0,1)

        #assert final_input_image.shape == (3,256,256) and final_input_image is not None

        return final_input_image, mask

    def __len__(self) -> int:
        return len(self.image_names)