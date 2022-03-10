from json.tool import main
import os
import tarfile 
import torch 
import torch.nn as nn 
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np 
from PIL import Image, ImageFilter
from data.custom_transforms import *
from data.utils import *  

class TrainPASCAL(data.Dataset):
    GOOGLE_DRIVE_ID = '1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'
    FILE = 'PASCAL_VOC.tgz'
    DB_NAME = 'VOCSegmentation'
    def __init__(self, root, split='trainaug', res=224, inv_list=[], eqv_list=[], \
        download=False):
        
        self.root  = root 
        self.split = split
        self.res = res
        self.inv_list = inv_list
        self.eqv_list = eqv_list
  
        
        if download:
            self._download()
        

        with open(os.path.join(self.root, self.DB_NAME, 'sets', '{}.txt'.format(self.split)), 'r') as f:
            lines = f.read().splitlines()
        
        print(len(lines))

        # self.reshuffle()
        _image_dir = os.path.join(self.root, self.DB_NAME, 'images')
        _sal_dir = os.path.join(self.root, self.DB_NAME, 'saliency_unsupervised_model')
        self.images = []
        self.sals = []
        for ii, line in enumerate(lines):
            _image = os.path.join(_image_dir, line + ".jpg")
            _sal = os.path.join(_sal_dir, line + ".png")
            if os.path.isfile(_image) and os.path.isfile(_sal):
                self.images.append(_image)
                self.sals.append(_sal)

        assert(len(self.images) == len(self.sals))
        
        self.reshuffle()
        print('Number of Images {}'.format(len(self.images)))

    def _download(self):
        
        _fpath = os.path.join(self.root, self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading dataset from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(os.path.join(self.root))
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def reshuffle(self):
        """
        Generate random floats for all image data to deterministically random transform.
        This is to use random sampling but have the same samples during clustering and 
        training within the same epoch. 
        """
        self.shuffled_indices = np.arange(len(self.images))
        np.random.shuffle(self.shuffled_indices)
        self.init_transforms() 

    def load_sal(self, index):
        return Image.open(self.sals[index])

    def load_data(self, index):
        """
        Labels are in unit8 format where class labels are in [0 - 181] and 255 is unlabeled.
        """
        return Image.open(self.images[index]).convert('RGB')
    
    def __getitem__(self, index):
        
        image = self.load_data(index)
        sal = self.load_sal(index)

        image, sal = self.transform_base(index, image, sal)

        image_query, sal_query = self.transform_image_sal(index, image, sal, ver=0)
        image_key, sal_key = self.transform_image_sal(index, image, sal, ver=1)
        
        label = self.get_pseudo_labels(index)

        return index, image_query, label, sal_query,  image_key, sal_key

    
    def get_pseudo_labels(self, index):
        if self.mode == 'label':
            label = torch.load(os.path.join(self.labeldir, 'label_query', '{}.pkl'.format(index)))
            label = torch.LongTensor(label)

            X1 = int(np.sqrt(label.shape[0]))
            
            label1 = label.view(X1, X1)

            return label1
        
        return None

    def transform_image_sal(self, index, image, sal, ver):

        image = self.transform_inv(index, image, ver=ver)
        if ver == 0: # Apply for just query
            image, sal = self.transform_eqv(index, image, sal)
          
        image, sal = self.transform_tensor(image, sal)
        sal = sal.squeeze().long()
        if len(sal.shape) == 3:
            sal = sal[0]
        return image, sal

    def transform_image(self, index, image):
        # Base transform
        image = self.transform_base(index, image)
        
        if self.mode == 'compute':
            if self.view == 1:
                image = self.transform_inv(index, image, 0)
                image = self.transform_tensor(image)
            elif self.view == 2:
                image = self.transform_inv(index, image, 1)
                image = TF.resize(image, self.res1, Image.BILINEAR)
                image = self.transform_tensor(image)
            else:
                raise ValueError('View [{}] is an invalid option.'.format(self.view))
            return (image, )
        elif 'train' in self.mode:
            # Invariance transform. 
            image1 = self.transform_inv(index, image, 0)
            image1 = self.transform_tensor(image1)

            if self.mode == 'baseline_train':
                return (image1, )
            
            image2 = self.transform_inv(index, image, 1)
            image2 = TF.resize(image2, self.res1, Image.BILINEAR)
            image2 = self.transform_tensor(image2)

            return (image1, image2)
        else:
            raise ValueError('Mode [{}] is an invalid option.'.format(self.mode))

    def transform_inv(self, index, image, ver):
        """
        Hyperparameters same as MoCo v2. 
        (https://github.com/facebookresearch/moco/blob/master/main_moco.py)
        """
        if 'brightness' in self.inv_list:
            image = self.random_color_brightness[ver](index, image)
        if 'contrast' in self.inv_list:
            image = self.random_color_contrast[ver](index, image)
        if 'saturation' in self.inv_list:
            image = self.random_color_saturation[ver](index, image)
        if 'hue' in self.inv_list:
            image = self.random_color_hue[ver](index, image)
        if 'gray' in self.inv_list:
            image = self.random_gray_scale[ver](index, image)
        if 'blur' in self.inv_list:
            image = self.random_gaussian_blur[ver](index, image)
        
        return image


    def transform_eqv(self, index, image, sal):
        
        if 'h_flip' in self.eqv_list:
            image, sal  = self.random_horizontal_flip(index, image, sal)
        if 'v_flip' in self.eqv_list:
            image, sal = self.random_vertical_flip(index, image, sal)

        return image, sal

    def init_transforms(self):
        N = len(self.images)
        # Base transform.
        self.transform_base = BaseTransform(self.res)
        
        # Transforms for invariance. 
        # Color jitter (4), gray scale, blur. 
        self.random_color_brightness = [RandomColorBrightness(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)]
        self.random_color_contrast   = [RandomColorContrast(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_saturation = [RandomColorSaturation(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_hue        = [RandomColorHue(x=0.1, p=0.8, N=N) for _ in range(2)]      # Control this later (NOTE)
        self.random_gray_scale    = [RandomGrayScale(p=0.2, N=N) for _ in range(2)]
        self.random_gaussian_blur = [RandomGaussianBlur(sigma=[.1, 2.], p=0.5, N=N) for _ in range(2)]
        
        # Transforms for equivariance
        self.random_horizontal_flip = RandomHorizontalTensorFlip(N=N)
        self.random_vertical_flip   = RandomVerticalFlip(N=N)

        # Tensor and normalize transform. 
        self.transform_tensor = TensorTransform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.images)


  
        

  
            
       
if __name__ == '__main__':
    inv_list = ['brightness', 'contrast', 'saturation', 'hue']
    eqv_list = ['h_flip', 'v_flip']
    trainset = TrainPASCAL('/home/khangt1k25/Code/Clustering-Segmentation/PASCAL_VOC', res=224, \
                        split='train', inv_list=inv_list, eqv_list=eqv_list) # NOTE: For now, max_scale = 1.  
    

    indice, img1, sal1, img2, sal2 = trainset[0]
    trainloader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=32,
                                                shuffle=True, 
                                                num_workers=2,
                                                pin_memory=True,
                                                # collate_fn=collate_train_baseline,
                                                worker_init_fn=worker_init_fn(2022))
    
    for i_batch, (indice, img1, sal1, img2, sal2) in enumerate(trainloader):
        print(indice.shape)
        print(indice)
        print(img1.shape)
        print(sal1.shape)
        print(img2.shape)
        print(sal2.shape)
        break
    # img1.show()
    # sal1.show()
    # img2.show()
    # sal2.show()