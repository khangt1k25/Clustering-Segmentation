from data.pascal_train_dataset import TrainPASCAL
from data.pascal_eval_dataset import EvalPASCAL
from utils import get_transform_params, collate_train, worker_init_fn
import torch
from commons import run_mini_batch_kmeans
from torchvision.transforms import ToPILImage

if __name__ == '__main__':
    # inv_list = ['blur', 'gray', 'brightness', 'contrast', 'saturation', 'hue']
    # eqv_list = ['h_flip', 'v_flip', 'random_crop']
    # inv_list = ['jiter', 'blur', 'gray']
    inv_list = ['gray']
    eqv_list = ['h_flip', 'random_crop']
    trainset = TrainPASCAL(root='./PASCAL_VOC/VOCSegmentation',
                            mode='compute',
                            split='trainaug',
                            labeldir='',
                            res1=224,
                            res2=224,
                            inv_list=inv_list,
                            eqv_list=eqv_list)
    evalset = EvalPASCAL(root='./PASCAL_VOC/VOCSegmentation', mode='test', split='val')
    

   
    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=32,
                                            shuffle=False, 
                                            num_workers=1,
                                            pin_memory=True,
                                            collate_fn=collate_train,
                                            worker_init_fn=worker_init_fn(16))

    toPIL = ToPILImage()

    trainloader.dataset.mode = 'compute'
    trainloader.dataset.reshuffle()
    trainloader.dataset.view = 1
    for i_batch, (indice, image) in enumerate(trainloader):
        
        print(image.shape)
        toPIL(image[0]).show()
        out = trainloader.dataset.transform_eqv(indice, image)

        print(out.shape)
        toPIL(out[0]).show()
        # feats = model(image)
        break
    
    #centroids1, kmloss1 = run_mini_batch_kmeans(args, logger, trainloader, model, view=1)
    #centroids2, kmloss2 = run_mini_batch_kmeans(args, logger, trainloader, model, view=2)