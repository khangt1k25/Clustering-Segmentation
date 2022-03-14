from functools import reduce
import os 
import numpy as np 
import torch 
import torch.nn as nn 

from scipy.optimize import linear_sum_assignment as linear_assignment

# from modules import fpn 
from utils import *

import warnings
warnings.filterwarnings('ignore')

def get_model_and_optimizer(args, logger, device):
    
    # Init model 
    
    from modules.builder import ContrastiveModel
    model = ContrastiveModel(args)
    model = model.to(device)
    
    classifier = initialize_classifier(args, split='train')
    classifier = classifier.to(device)

    # Init optimizer 
    if args.optim_type == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, \
                                    momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    elif args.optim_type == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, nesterov=False)

    # optional restart. 
    args.start_epoch  = 0 
    if args.restart or args.eval_only: 
        load_path = os.path.join(args.save_model_path, 'checkpoint.pth.tar')
        if args.eval_only:
            load_path = args.eval_path
        if os.path.isfile(load_path):
            checkpoint  = torch.load(load_path)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            classifier.load_state_dict(checkpoint['classifier1_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('Loaded checkpoint. [epoch {}]'.format(args.start_epoch))
        else:
            logger.info('No checkpoint found at [{}].\nStart from beginning...\n'.format(load_path))
    
    return model, optimizer, classifier







def get_optimizer(args, parameters):
    # Init optimizer 
    if args.optim_type == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, parameters), lr=args.lr, \
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim_type == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, parameters), lr=args.lr)
    
    return optimizer


def run_mini_batch_kmeans(args, logger, dataloader, model, device, split='train'):
    '''
    Clustering for Key view
    '''
    kmeans_loss  = AverageMeter('kmean loss')
    faiss_module = get_faiss_module(args)
    
    if split=='train':
        K = args.K_train
    elif split == 'test':
        K = args.K_test
    
    data_count   = np.zeros(K)
    featslist    = []
    num_batches  = 0
    first_batch  = True    
    isreduce = (args.reducer > 0)
    
    model.eval()
    with torch.no_grad():
        for i_batch, (_, _, _, _, img_k, sal_k) in enumerate(dataloader):
            
            img_k, sal_k = img_k.cuda(non_blocking=True), sal_k.cuda(non_blocking=True)
            k, _ = model.model_k(img_k) # Bx dim x H x W
            k = nn.functional.normalize(k, dim=1)
            batch_size = k.shape[0]
            k = k.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
            k = torch.reshape(k, [-1, args.ndim]) # queries: BHW x dim
            
            # Drop background pixels
            offset = torch.arange(0, 2 * batch_size, 2).to(sal_k.device)
            sal_k = (sal_k + torch.reshape(offset, [-1, 1, 1]))*sal_k 
            sal_k = sal_k.view(-1)
            mask_indexes = torch.nonzero((sal_k)).view(-1).squeeze()
    
            if isreduce:
                reducer_idx = torch.randperm(mask_indexes.shape[0])[:args.reducer*batch_size]
                mask_indexes = mask_indexes[reducer_idx]
            
            k = torch.index_select(k, index=mask_indexes, dim=0).detach().cpu() 
            

            if i_batch == 0:
                logger.info('Batch feature : {}'.format(list(k.shape)))
            
            if num_batches < args.num_init_batches:
                featslist.append(k)
                num_batches += 1
                if num_batches == args.num_init_batches or num_batches == len(dataloader):
                    if first_batch:
                        # Compute initial centroids. 
                        # By doing so, we avoid empty cluster problem from mini-batch K-Means. 
                        featslist = torch.cat(featslist).cpu().numpy().astype('float32')
                        centroids = get_init_centroids(args, K, featslist, faiss_module).astype('float32')
                        D, I = faiss_module.search(featslist, 1)
                        kmeans_loss.update(D.mean())
                        logger.info('Initial k-means loss: {:.4f} '.format(kmeans_loss.avg))
                        # Compute counts for each cluster. 
                        for k in np.unique(I):
                            data_count[k] += len(np.where(I == k)[0])
                        first_batch = False

                        # break # discard this 
                    else:
                        b_feat = torch.cat(featslist)
                        faiss_module = module_update_centroids(faiss_module, centroids)
                        D, I = faiss_module.search(b_feat.numpy().astype('float32'), 1)
                        kmeans_loss.update(D.mean())

                        # Update centroids. 
                        for k in np.unique(I):
                            idx_k = np.where(I == k)[0]
                            data_count[k] += len(idx_k)
                            centroid_lr    = len(idx_k) / (data_count[k] + 1e-6)
                            centroids[k]   = (1 - centroid_lr) * centroids[k] + centroid_lr * b_feat[idx_k].mean(0).numpy().astype('float32')
                    
                    # Empty. 
                    featslist   = []
                    num_batches = args.num_init_batches - args.num_batches

            if (i_batch % 100) == 0:
                logger.info('[Saving features]: {} / {} | [K-Means Loss]: {:.4f}'.format(i_batch, len(dataloader), kmeans_loss.avg))
    
    del faiss_module
    centroids = torch.tensor(centroids, requires_grad=False).to(device)

    return centroids, kmeans_loss.avg




def compute_labels(args, logger, dataloader, model, centroids, device):
    """
    Label for Query view
    The distance is efficiently computed by setting centroids as convolution layer. 
    """
    K = centroids.size(0) + 1

    # Define metric function with conv layer. 
    metric_function = get_metric_as_conv(centroids, device)
    counts = torch.zeros(K, requires_grad=False).cpu()
    model.eval()
    with torch.no_grad():
        for i_batch, (indice, img_q, sal_q, _, _, _) in enumerate(dataloader):
            img_q, sal_q = img_q.cuda(non_blocking=True), sal_q.cuda(non_blocking=True)
            q, _ = model.model_q(img_q) # Bx dim x H x W
            q = nn.functional.normalize(q, dim=1)

            if i_batch == 0:
                logger.info('Centroid size      : {}'.format(list(centroids.shape)))
                logger.info('Batch input size   : {}'.format(list(img_q.shape)))
                logger.info('Batch feature size : {}\n'.format(list(q.shape)))

            # Compute distance and assign label. 
            scores  = compute_negative_euclidean(q, centroids, metric_function) #BxCxHxW: all bg 're 0 

            # Save labels and count. 
            for idx, idx_img in enumerate(indice):
                counts += postprocess_label(args, K, idx, idx_img, scores, sal_q, view='query')
            
            if (i_batch % 200) == 0:
                logger.info('[Assigning labels] {} / {}'.format(i_batch, len(dataloader)))
    
    weight = counts / counts.sum()
        
    return weight


def evaluate(args, logger, dataloader, model, classifier, device):
    histogram = np.zeros((args.K_test, args.K_test))
        
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for i_batch, (indice, img, sal, label) in enumerate(dataloader):
            img, sal = img.to(device), sal.to(device)
            q, _ = model.model_q(img)
            q = nn.functional.normalize(q, dim=1) # BxdimxHxW
            probs = classifier(q) #BxdimxHxW
            probs = F.interpolate(probs, label.shape[-2:], mode='bilinear', align_corners=False)
            
            preds = probs.topk(1, dim=1)[1].squeeze().view(-1).cpu().numpy()

            label = label.view(-1).cpu().numpy()

            
            valid_preds = preds[label != 255]
            valid_label = label[label != 255]

            valid_preds = valid_preds[valid_label != 0]
            valid_label = valid_label[valid_label != 0]
            valid_label = valid_label - 1


            histogram += scores(valid_label, valid_preds, args.K_test)
            
            if i_batch%20==0:
                logger.info('{}/{}'.format(i_batch, len(dataloader)))
    
    # Hungarian Matching. 
    m = linear_assignment(histogram.max() - histogram)
    
    # Evaluate. 
    match = np.array(list(zip(*m)))
    acc = histogram[match[:, 0], match[:, 1]].sum() / histogram.sum() * 100

    new_hist = np.zeros((args.K_test, args.K_test))
    for idx in range(args.K_test):
        new_hist[match[idx, 1]] = histogram[idx]
    

    res = get_result_metrics(new_hist)

    return acc, res
