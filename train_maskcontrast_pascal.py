import argparse
from operator import inv
import os
from posixpath import split
import time as t
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.pascal_train_dataset import TrainPASCAL
from data.pascal_eval_dataset import EvalPASCAL
from utils import *
from commons import *
from modules import fpn 
from torch.utils.tensorboard import SummaryWriter, writer 

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--restart_path', type=str)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed for reproducability.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers.')
    parser.add_argument('--restart', action='store_true', default=True)
    parser.add_argument('--num_epoch', type=int, default=60) 

    
    # Model 
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--pretraining', type=str, default='imagenet_classification')
    parser.add_argument('--moco_state_dict', type=str, default='/content/drive/MyDrive/UCS_local/moco_v2_800ep_pretrain.pth.tar')
    parser.add_argument('--ndim', type=int, default=32)

    # Optimizer
    parser.add_argument('--optim_type', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_scheduler', type=str, default='poly')
    
    # Train. 
    parser.add_argument('--batch_size_cluster', type=int, default=32)
    parser.add_argument('--batch_size_train', type=int, default=32)
    parser.add_argument('--batch_size_test', type=int, default=32)
    
    parser.add_argument('--num_init_batches', type=int, default=64)
    parser.add_argument('--num_batches', type=int, default=2)
    parser.add_argument('--kmeans_n_iter', type=int, default=20)
    
    parser.add_argument('--eval_interval', type=int, default=10)

    # Cluster 
    parser.add_argument('--K_train', type=int, default=20)
    parser.add_argument('--K_test', type=int, default=20) 
    parser.add_argument('--reducer', type=int, default=0)
    parser.add_argument('--coeff', type=float, default=1e-1)


    # Dataset. 
    parser.add_argument('--res', type=int, default=224, help='Input size.')
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--equiv', action='store_true', default=False)
    parser.add_argument('--jitter', action='store_true', default=False)
    parser.add_argument('--grey', action='store_true', default=False)
    parser.add_argument('--blur', action='store_true', default=False)
    parser.add_argument('--h_flip', action='store_true', default=False)
    parser.add_argument('--v_flip', action='store_true', default=False)
    
    # Eval-only
    parser.add_argument('--repeats', type=int, default=5) 
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str)

    return parser.parse_args()



def train(args, logger, dataloader, model, optimizer, device, epoch):
    losses = AverageMeter('Total loss')
    contrastive_losses = AverageMeter('Contrastive Loss')
    cluster_losses = AverageMeter('Cluster loss')
    saliency_losses = AverageMeter('Saliency Loss')
    progress = ProgressMeter(len(dataloader), 
                        [losses, contrastive_losses , cluster_losses, saliency_losses],
                        prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    for i_batch, (_, img_q, sal_q, _, img_k, sal_k) in enumerate(dataloader):
        
        img_q = img_q.cuda(non_blocking=True)
        sal_q = sal_q.cuda(non_blocking=True)
        img_k = img_k.cuda(non_blocking=True)
        sal_k = sal_k.cuda(non_blocking=True) 

        logits, labels, saliency_loss = model.mc_forward(img_q, sal_q, img_k, sal_k)

         # Use E-Net weighting for calculating the pixel-wise loss.
        uniq, freq = torch.unique(labels, return_counts=True)
        p_class = torch.zeros(logits.shape[1], dtype=torch.float32).cuda(non_blocking=True)
        p_class_non_zero_classes = freq.float() / labels.numel()
        p_class[uniq] = p_class_non_zero_classes
        w_class = 1 / torch.log(1.02 + p_class)
        contrastive_loss = F.cross_entropy(logits, labels, weight=w_class,
                                            reduction='mean')


        
        loss = contrastive_loss + saliency_loss 


        contrastive_losses.update(contrastive_loss.item())
        saliency_losses.update(saliency_loss.item())
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Display progress
        if i_batch % 25 == 0:
            progress.display(i_batch)
    
    
    
    writer_path = os.path.join(args.save_model_path, "runs")
    writer = SummaryWriter(log_dir=writer_path)
    writer.add_scalar('total loss', losses.avg, epoch)
    writer.add_scalar('contrastive loss', contrastive_losses.avg, epoch)
    writer.add_scalar('saliency loss', saliency_losses.avg, epoch)
    writer.close()

    return losses.avg



def main(args, logger):
    logger.info(args)

    # Use random seed.
    fix_seed_for_reproducability(args.seed)

    # Start time.
    t_start = t.time()
    device = torch.device('cuda' if torch.cuda.is_available() else'cpu' )

    # Get model and optimizer.
    model, optimizer = get_model_and_optimizer(args, logger, device)

    # Dataset
    inv_list, eqv_list = get_transform_params(args)
    trainset = TrainPASCAL(args.data_root, res=args.res, split='trainaug', inv_list=inv_list, eqv_list=eqv_list)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=args.batch_size_train,
                                                shuffle=True, 
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                worker_init_fn=worker_init_fn(args.seed),
                                                drop_last=True,
                                                )

    testset    = EvalPASCAL(args.data_root, res=args.res, split='val', transform_list=[])
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size_test,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             worker_init_fn=worker_init_fn(args.seed),
                                             )

    
    
    # Train start.
    for epoch in range(args.start_epoch, args.num_epoch):
        #  Clustering
        logger.info('\n============================= [Epoch {}] =============================\n'.format(epoch))

    
        logger.info('Start training ...\n')
        
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Adjusted learning rate to {:.5f} \n'.format(lr))
        
        t2 = t.time()
        train_loss = train(args, logger, trainloader, model, optimizer, device, epoch) 
        trainset.mode  = 'normal'
        logger.info('  Training loss  : {:.5f}.\n'.format(train_loss))
        logger.info('Finish training ...\n')

        ## Evaluating
        # if epoch% args.eval_interval == 0:
        #     logger.info('Start evaluating ...\n')

        #     centroids, kmloss = run_mini_batch_kmeans_for_testloader(args, logger, evalloader, model, device)
            
        #     classifier = initialize_classifier(args, split='test')
        #     classifier = classifier.to(device)
        #     classifier.weight.data = centroids.unsqueeze(-1).unsqueeze(-1)
        #     freeze_all(classifier)
        #     del centroids
        
        #     acc, res   = evaluate(args, logger, testloader, model, classifier, device)
            

        #     logger.info('========== Evaluatation at epoch [{}] ===========\n'.format(epoch))
        #     logger.info('  Time for train/eval : [{}].\n'.format(get_datetime(int(t.time())-int(t2))))
        #     logger.info('  ACC: {:.4f} | mIoU: {:.4f} | mean_precision {:.4f} | overall_precision {:.4f} \n'.format(acc, res['mean_iou'], res['mean_precision (class-avg accuracy)'],res['overall_precision (pixel accuracy)']))
        #     logger.info('=================================================\n')
        #     logger.info('Finish evaluating ...\n')

        logger.info('Start checkpointing ...\n')
        torch.save({'epoch': epoch+1, 
                    'args' : args,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    },
                    os.path.join(args.save_model_path, 'checkpoint.pth.tar'))
        logger.info('Finish checkpointing ...\n')
    
        
              
    # Evaluate with fresh clusters. 
    acc_list_new = []  
    res_list_new = []
    logger.info('================================Start evaluating the LAST==============================\n')                 
    
    evalset = EvalPASCAL(args.data_root, res=args.res, split='val', transform_list=['jiter', 'grey'])
    evalloader = torch.utils.data.DataLoader(evalset, 
                                            batch_size=args.batch_size_cluster,
                                            shuffle=True, 
                                            num_workers=args.num_workers,
                                            pin_memory=True,
                                            worker_init_fn=worker_init_fn(args.seed),
                                            )
    if args.repeats > 0:
        for r in range(args.repeats):
            logger.info('============ Start Repeat Time {}============\n'.format(r))                 
            t1 = t.time()
            logger.info('Start clustering \n')
            centroids, kmloss = run_mini_batch_kmeans_for_testloader(args, logger, evalloader, model, device)
            logger.info('Finish clustering with [Loss: {:.5f}/ Time: {}]\n'.format(kmloss, get_datetime(int(t.time())-int(t1))))
            
            
            classifier = initialize_classifier(args, split='test')
            classifier = classifier.to(device)
            classifier.weight.data = centroids.unsqueeze(-1).unsqueeze(-1)
            freeze_all(classifier)
            
            acc_new, res_new = evaluate(args, logger, testloader, model, classifier, device)
            acc_list_new.append(acc_new)
            res_list_new.append(res_new)     
            logger.info('  ACC: {:.4f} | mIoU: {:.4f} \n'.format(acc_new, res_new['mean_iou']))
            logger.info('============Finish Repeat Time {}============\n'.format(r)) 

    else:
        logger.info('Repeats must be positive')

    logger.info('Average overall pixel accuracy [NEW] : {} +/- {}.'.format(round(np.mean(acc_list_new), 2), np.std(acc_list_new)))
    logger.info('Average mIoU [NEW] : {:.3f} +/- {:.3f}. '.format(np.mean([res['mean_iou'] for res in res_list_new]), 
                                                                  np.std([res['mean_iou'] for res in res_list_new])))
    logger.info('================================Finish evaluating the LAST==============================\n')
    
    logger.info('Experiment done. [{}]\n'.format(get_datetime(int(t.time())-int(t_start))))
    
if __name__=='__main__':
    args = parse_arguments()

    # Setup the path to save.
    if not args.pretrain:
        args.save_root += '/scratch'
    if args.augment:
        args.save_root += '/augmented/res={}/jitter={}_blur={}_grey={}'.format(args.res, args.jitter, args.blur, args.grey)
    if args.equiv:
        args.save_root += '/equiv/h_flip={}_v_flip={}'.format(args.h_flip, args.v_flip)

    args.save_model_path = os.path.join(args.save_root, args.comment, 'K_train={}'.format(args.K_train))
    args.save_eval_path  = os.path.join(args.save_model_path, 'K_test={}'.format(args.K_test))
    
    if not os.path.exists(args.save_eval_path):
        os.makedirs(args.save_eval_path)

    # Setup logger.
    logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
    
    # Start.
    main(args, logger)


 

