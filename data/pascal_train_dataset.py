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
from data.v2 import RandAugment2
# from custom_transforms import *
# from utils import *
# from v2 import RandAugment2

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
        self.mode = 'normal'
        
        if download:
            self._download()
        
        with open(os.path.join(self.root, self.DB_NAME, 'sets', '{}.txt'.format(self.split)), 'r') as f:
            lines = f.read().splitlines()
        

        # self.reshuffle()
        _image_dir = os.path.join(self.root, self.DB_NAME, 'images')
        _sal_dir = os.path.join(self.root, self.DB_NAME, 'saliency_unsupervised_model')
        self.images = []
        self.sals = []
        self.names = []
        for ii, line in enumerate(lines):
            _image = os.path.join(_image_dir, line + ".jpg")
            _sal = os.path.join(_sal_dir, line + ".png")
            if os.path.isfile(_image) and os.path.isfile(_sal):
                self.images.append(_image)
                self.sals.append(_sal)
                self.names.append(line)
        
        assert(len(self.images) == len(self.sals))
        
        print('Number of Images {}'.format(len(self.images)))

        self.init_transforms()

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


    def load_sal(self, index):
        return Image.open(self.sals[index])

    def load_data(self, index):
        return Image.open(self.images[index]).convert('RGB')
    
    def __getitem__(self, index):
        
        # name = self.names[index]

        image = self.load_data(index)
        sal = self.load_sal(index)

        image, sal = self.transform_base(index, image, sal)
   

        image_query, sal_query = self.transform_image_sal(index, image, sal, ver=0)
        image_key, sal_key = self.transform_image_sal(index, image, sal, ver=1)
        
        image_randaug, sal_randaug = self.randAugment(index, image, sal)

        # label_query = self.get_pseudo_labels(index)
        
        return index, image_query, sal_query, image_key, sal_key, image_randaug, sal_randaug


    def get_pseudo_labels(self, index):
        if self.mode == 'label':
            label = torch.load(os.path.join(self.labeldir, 'label_query', '{}.pkl'.format(index)))
            label = torch.LongTensor(label)

            X1 = int(np.sqrt(label.shape[0]))
            
            label1 = label.view(X1, X1)

            return label1
        
        return torch.zeros([])

    def transform_image_sal(self, index, image, sal, ver):

        image = self.transform_inv(index, image, ver=ver)
        if ver == 0: # Apply for just query
            image, sal = self.transform_eqv(index, image, sal)
          
        image, sal = self.transform_tensor(image, sal)
        sal = sal.squeeze().long()
        if len(sal.shape) == 3:
            sal = sal[0]
        
        return image, sal

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

    
    def transform_eqv_repr(self, index, feat):
        if 'h_flip' in self.eqv_list:
            feat = self.horizontal_tensor_flip(index, feat)
        if 'v_flip' in self.eqv_list:
            feat = self.vertical_tensor_flip(index, feat)

        return feat
    
    def transform_ranaug_repr(self, index, feat, sal):
        return self.randAugment(indice, feat, sal)
    
    def init_transforms(self):
        N = len(self.images)
        # Base transform.
        self.transform_base = RandomResizedCrop(size=self.res, scale=(0.2, 1)) 
        
        # Transforms for invariance. 
        # Color jitter (4), gray scale, blur. 
        self.random_color_brightness = [RandomColorBrightness(x=0.4, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)]
        self.random_color_contrast   = [RandomColorContrast(x=0.4, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_saturation = [RandomColorSaturation(x=0.4, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_hue        = [RandomColorHue(x=0.1, p=0.8, N=N) for _ in range(2)]      # Control this later (NOTE)
        
        self.random_gray_scale    = [RandomGrayScale(p=0.2, N=N) for _ in range(2)]

        self.random_gaussian_blur = [RandomGaussianBlur(sigma=[.1, 2.], p=0.5, N=N) for _ in range(2)]
        
        # Transforms for equivariance
        self.random_horizontal_flip = RandomHorizontalFlip(N=N)
        self.random_vertical_flip   = RandomVerticalFlip(N=N)

        # For features
        self.horizontal_tensor_flip = RandomHorizontalTensorFlip(N=N, p_ref=self.random_horizontal_flip.p_ref, plist=self.random_horizontal_flip.plist)
        self.vertical_tensor_flip = RandomVerticalTensorFlip(N=N, p_ref=self.random_vertical_flip.p_ref, plist=self.random_vertical_flip.plist)


        # RandAugment
        self.randAugment = RandAugment2(N=N, k=10, m=10)


        # Tensor and normalize transform. 
        self.transform_tensor = TensorTransform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.images)


  
        

  
            
       
if __name__ == '__main__':
    inv_list = ['brightness', 'contrast', 'saturation', 'hue', 'gray', 'blur']
    eqv_list = ['h_flip', 'v_flip']
    trainset = TrainPASCAL('/home/khangt1k25/Code/Clustering-Segmentation/PASCAL_VOC', res=224, \
                        split='train', inv_list=inv_list, eqv_list=eqv_list) # NOTE: For now, max_scale = 1.  
    
    # i = np.random.randint(0, 100)
    i = 100
    indice, img1, sal1, img2, sal2, label, name, randaug, randsal = trainset[i]
    trainloader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=32,
                                                shuffle=True, 
                                                num_workers=2,
                                                pin_memory=True,
    )
    
    
    
    

                                                # collate_fn=collate_train_baseline,
                                                # worker_init_fn=worker_init_fn(2022))
    
    topil = torchvision.transforms.ToPILImage()

    # topil(randaug).show()
    randaug.show()
    randsal.show()
    # for i_batch, (indice, img1, sal1, img2, sal2, label, name, randaug) in enumerate(trainloader):
    #     # print(img1.shape)
    #     # print(sal1.shape)
    #     # print(img2.shape)
    #     # print(sal2.shape)

    #     # print(indice.shape)
    #     print(name)

    #     topil(img1[1]).show()
    #     # topil(img2[0]).show()
    #     feat3 = trainloader.dataset.transform_eqv_repr(indice, img2)
    #     topil(feat3[1]).show()
    #     # if i_batch==10:
    #     #     break
    #     break
        



    