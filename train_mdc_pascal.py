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
    parser.add_argument('--num_epoch', type=int, default=10) 
    parser.add_argument('--repeats', type=int, default=5)  

    # Train. 
    parser.add_argument('--res', type=int, default=224, help='Input size.')
    parser.add_argument('--batch_size_cluster', type=int, default=32)
    parser.add_argument('--batch_size_train', type=int, default=32)
    parser.add_argument('--batch_size_test', type=int, default=32)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--num_init_batches', type=int, default=64)
    parser.add_argument('--num_batches', type=int, default=64)
    parser.add_argument('--kmeans_n_iter', type=int, default=30)

    # Additonal
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--pretraining', type=str, default='imagenet_classification')
    parser.add_argument('--moco_state_dict', type=str, default='/content/drive/MyDrive/UCS_local(renamed)/moco_v2_800ep_pretrain.pth.tar')
    parser.add_argument('--ndim', type=int, default=32)
    parser.add_argument('--reducer', type=int, default=0)
    parser.add_argument('--eval_interval', type=int, default=1)


    # Loss. 
    parser.add_argument('--K_train', type=int, default=1000)
    parser.add_argument('--K_test', type=int, default=20) 

    # Dataset. 
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--equiv', action='store_true', default=False)
    parser.add_argument('--min_scale', type=float, default=0.5)
    parser.add_argument('--jitter', action='store_true', default=False)
    parser.add_argument('--grey', action='store_true', default=False)
    parser.add_argument('--blur', action='store_true', default=False)
    parser.add_argument('--h_flip', action='store_true', default=False)
    parser.add_argument('--v_flip', action='store_true', default=False)
    parser.add_argument('--random_crop', action='store_true', default=False)
    parser.add_argument('--val_type', type=str, default='val')
    
    # Eval-only
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str)

    return parser.parse_args()


def train(args, logger, dataloader, model, classifier, optimizer, device, epoch, kmloss):
    losses = AverageMeter('Total loss')
    contrastive_losses = AverageMeter('Contrastive Loss')
    cluster_losses = AverageMeter('Cluster loss')
    saliency_losses = AverageMeter('Saliency Loss')
    progress = ProgressMeter(len(dataloader), 
                        [losses, contrastive_losses , cluster_losses, saliency_losses],
                        prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    for i_batch, (indice, img_q, sal_q, label, img_k, sal_k) in enumerate(dataloader):
        
        img_q = img_q.cuda(non_blocking=True)
        sal_q = sal_q.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        img_k = img_k.cuda(non_blocking=True)
        sal_k = sal_k.cuda(non_blocking=True)

        
        logits, labels, cluster_logits, cluster_labels, saliency_loss = model(img_q, sal_q, img_k, sal_k, classifier, label)

         # Use E-Net weighting for calculating the pixel-wise loss.
        uniq, freq = torch.unique(labels, return_counts=True)
        p_class = torch.zeros(logits.shape[1], dtype=torch.float32).cuda(p['gpu'], non_blocking=True)
        p_class_non_zero_classes = freq.float() / labels.numel()
        p_class[uniq] = p_class_non_zero_classes
        w_class = 1 / torch.log(1.02 + p_class)
        contrastive_loss = F.cross_entropy(logits, labels, weight=w_class,
                                            reduction='mean')

        cluster_loss = F.cross_entropy(cluster_logits, cluster_labels, reduction='mean')

        
        lamda = 0.1
        loss = contrastive_loss + saliency_loss + 0.1*cluster_loss


        contrastive_losses.update(contrastive_loss.item())
        cluster_losses.update(cluster_loss.item())
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
    writer.add_scalar('cluster loss', cluster_losses.avg, epoch)
    writer.add_scalar('saliency loss', saliency_losses.avg, epoch)
    writer.add_scalar('kmeans loss', kmloss, epoch)
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
    model, optimizer, classifier = get_model_and_optimizer(args, logger, device)

    # New trainset inside for-loop.
    inv_list, eqv_list = get_transform_params(args)
    trainset = TrainPASCAL(args.data_root, res=args.res, split='trainaug', inv_list=inv_list, eqv_list=eqv_list) # NOTE: For now, max_scale = 1.
    trainloader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=args.batch_size_cluster,
                                                shuffle=True, 
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                worker_init_fn=worker_init_fn(args.seed),
                                                # drop_last=True
                                                )

    testset    = EvalPASCAL(args.data_root, res=args.res, split='val', transform_list=['jitter', 'blur', 'grey'])
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size_test,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             worker_init_fn=worker_init_fn(args.seed),
                                            #  drop_last=True
                                             )


    
    # Train start.
    for epoch in range(args.start_epoch, args.num_epoch):
        # Assign probs. 
        logger.info('\n============================= [Epoch {}] =============================\n'.format(epoch))
        logger.info('Start clustering ... \n')
        t1 = t.time()
        
        centroids, kmloss = run_mini_batch_kmeans(args, logger, trainloader, model, device=device, split='train')
        
        logger.info('Finish clustering with loss {} and time: [{}]\n'.format(kmloss ,get_datetime(int(t.time())-int(t1))))
        
        ## Compute cluster assignment. 
        t2 = t.time()
        weight = compute_labels(args, logger, trainloader, model, centroids, device=device) 
        
        # np.save('/content/drive/MyDrive/UCS_local_2/Clustering-Segmentation/results/weight.npy', weight.cpu().numpy())

        logger.info('-Cluster labels ready. [{}]\n'.format(get_datetime(int(t.time())-int(t2)))) 

        ## Criterion. 
        # criterion = torch.nn.CrossEntropyLoss(weight=weight).cuda()
        # criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)


        ## Set nonparametric classifier.
        classifier = initialize_classifier(args)
        classifier = classifier.to(device)
        classifier.weight.data = centroids.unsqueeze(-1).unsqueeze(-1)
        freeze_all(classifier)
        del centroids

        ## Set trainset to get pseudolabel
        trainset.mode  = 'label'
        trainset.labeldir = args.save_model_path
        trainloader_loop  = torch.utils.data.DataLoader(trainset, 
                                                        batch_size=args.batch_size_train, 
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        # collate_fn=collate_train_baseline,
                                                        worker_init_fn=worker_init_fn(args.seed),
                                                        drop_last=True,
                                                        )


        logger.info('Start training ...\n')
        t2 = t.time()
        train_loss = train(args, logger, trainloader_loop, model, classifier, optimizer, device, epoch, kmloss) 
        logger.info('Finish training ...\n')

        if (args.K_train == args.K_test) and (epoch% args.eval_interval == 0):
            logger.info('Start evaluating ...\n')
            acc, res   = evaluate(args, logger, testloader, model, classifier, device)
            logger.info('========== Evaluatation at epoch [{}] ===========\n'.format(epoch))
            logger.info('  Time for train/eval : [{}].\n'.format(get_datetime(int(t.time())-int(t2))))
            logger.info('  K-Means loss   : {:.5f}.\n'.format(kmloss))
            logger.info('  Training loss  : {:.5f}.\n'.format(train_loss))
            logger.info('  ACC: {:.4f} | mIoU: {:.4f} | mean_precision {:.4f} | overall_precision {:.4f} \n'.format(acc, res['mean_iou'], res['mean_precision (class-avg accuracy)'],res['overall_precision (pixel accuracy)']))
            logger.info('=================================================\n')
            logger.info('Finish evaluating ...\n')

        logger.info('Start checkpointing ...\n')
       
        torch.save({'epoch': epoch+1, 
                    'args' : args,
                    'state_dict': model.state_dict(),
                    'classifier1_state_dict' : classifier.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    },
                    os.path.join(args.save_model_path, 'checkpoint.pth.tar'))
        logger.info('Finish checkpointing ...\n')

        
         

    ## Evaluate.
    # We do not use this 
    # trainset    = EvalPASCAL(args.data_root, res=args.res, split='train')
    # trainloader = torch.utils.data.DataLoader(trainset, 
    #                                             batch_size=args.batch_size_cluster,
    #                                             shuffle=True,
    #                                             num_workers=args.num_workers,
    #                                             pin_memory=True,
    #                                             worker_init_fn=worker_init_fn(args.seed),
    #                                             drop_last=True)

    # testset    = EvalPASCAL(args.data_root, res=args.res, split='val')
    # testloader = torch.utils.data.DataLoader(testset, 
    #                                          batch_size=args.batch_size_test,
    #                                          shuffle=False,
    #                                          num_workers=args.num_workers,
    #                                          pin_memory=True,
    #                                          worker_init_fn=worker_init_fn(args.seed),
    #                                          drop_last=True)
    
                          
    # Evaluate with fresh clusters. 
    acc_list_new = []  
    res_list_new = []
    logger.info('================================Start evaluating the LAST==============================\n')                 
    
    
    if args.repeats > 0:
        for r in range(args.repeats):
            logger.info('============ Start Repeat Time {}============\n'.format(r))                 
            t1 = t.time()
            logger.info('Start clustering \n')
            centroids, kmloss = run_mini_batch_kmeans(args, logger, trainloader, model, device, split='test')
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
        logger.info('============ Using trained cluster============\n')
        acc_new, res_new = evaluate(args, logger, testloader, classifier, model)
        acc_list_new.append(acc_new)
        res_list_new.append(res_new)
    

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
        args.save_root += '/equiv/h_flip={}_v_flip={}_crop={}/'.format(args.h_flip, args.v_flip, args.random_crop)

    args.save_model_path = os.path.join(args.save_root, args.comment, 'K_train={}'.format(args.K_train))
    args.save_eval_path  = os.path.join(args.save_model_path, 'K_test={}'.format(args.K_test))
    
    if not os.path.exists(args.save_eval_path):
        os.makedirs(args.save_eval_path)

    # Setup logger.
    logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
    
    # Start.
    main(args, logger)


 

