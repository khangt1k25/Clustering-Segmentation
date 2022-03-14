from data.pascal_train_dataset import TrainPASCAL
from data.pascal_eval_dataset import EvalPASCAL
# from utils import get_transform_params, collate_train, worker_init_fn, collate_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import numpy as np
import matplotlib.pyplot as plt


VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


if __name__ == '__main__':

    inv_list = ['jiter', 'blur', 'gray']
    eqv_list = ['h_flip', 'v_flip']
 

    evalset = EvalPASCAL(root='./PASCAL_VOC', split='val', res=224)
    

   



    testloader = torch.utils.data.DataLoader(evalset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=1,
                                             pin_memory=True,)
    toPIL = ToPILImage()
    for i, (_, image, label) in enumerate(testloader):
        print(image.shape)
        print(label.shape)
        print(torch.unique(label[0]))
        toPIL(image[0].squeeze(0)).show()
        toPIL(label[0].float()).show()
        # toPIL(sal[0].float()).show()
        break


    