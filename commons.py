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
    
    classifier = initialize_classifier(args)
    classifier = classifier.to(device)

    # Init optimizer 
    if args.optim_type == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, \
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim_type == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)

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


def run_mini_batch_kmeans(args, logger, dataloader, model, device):
    """
    num_init_batches: (int) The number of batches/iterations to accumulate before the initial k-means clustering.
    num_batches     : (int) The number of batches/iterations to accumulate before the next update. 
    """
    kmeans_loss  = AverageMeter('kmean loss')
    faiss_module = get_faiss_module(args)
    data_count   = np.zeros(args.K_train)
    featslist    = []
    num_batches  = 0
    reducer = 100
    first_batch  = True
    drop = True

    model.eval()
    with torch.no_grad():
        for i_batch, (indice, _, _, img_k, sal_k) in enumerate(dataloader):
            
            # img_k, sal_k = img_k.to(device), sal_k.to(device)
            img_k, sal_k = img_k.cuda(non_blocking=True), sal_k.cuda(non_blocking=True)
            k, _ = model.model_k(img_k) # Bx dim x H x W
            k = nn.functional.normalize(k, dim=1)
            batch_size, dim = k.shape[0], k.shape[1]
            k = k.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
            k = torch.reshape(k, [-1, dim]) # queries: BHW x dim
            
            # Drop background pixels
            if drop:
                offset = torch.arange(0, 2 * batch_size, 2).to(sal_k.device)
                sal_k = (sal_k + torch.reshape(offset, [-1, 1, 1]))*sal_k 
                sal_k = sal_k.view(-1)
                
                mask_indexes = torch.nonzero((sal_k)).view(-1).squeeze()
                reducer_idx = torch.randperm(mask_indexes.shape[0])[:reducer*batch_size]
                mask_indexes = mask_indexes[reducer_idx]
                k = torch.index_select(k, index=mask_indexes, dim=0).detach().cpu() # pixels x dim 
            else:
                k = k.detach().cpu()

            if i_batch == 0:
                logger.info('Batch input size : {}'.format(list(img_k.shape)))
                logger.info('Batch feature : {}'.format(list(k.shape)))
            
            # feats = feature_flatten(feats).detach().cpu()
            if num_batches < args.num_init_batches:
                featslist.append(k)
                num_batches += 1
                
                if num_batches == args.num_init_batches or num_batches == len(dataloader):
                    if first_batch:
                        # Compute initial centroids. 
                        # By doing so, we avoid empty cluster problem from mini-batch K-Means. 
                        featslist = torch.cat(featslist).cpu().numpy().astype('float32')
                        centroids = get_init_centroids(args, args.K_train, featslist, faiss_module).astype('float32')
                        D, I = faiss_module.search(featslist, 1)

                        kmeans_loss.update(D.mean())
                        logger.info('Initial k-means loss: {:.4f} '.format(kmeans_loss.avg))
                        
                        # Compute counts for each cluster. 
                        for k in np.unique(I):
                            data_count[k] += len(np.where(I == k)[0])
                        first_batch = False
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

    centroids = torch.tensor(centroids, requires_grad=False).to(device)

    return centroids, kmeans_loss.avg




def compute_labels(args, logger, dataloader, model, centroids, device):
    """
    Label all images for each view with the obtained cluster centroids. 
    The distance is efficiently computed by setting centroids as convolution layer. 
    """
    K = centroids.size(0)

    # Define metric function with conv layer. 
    metric_function = get_metric_as_conv(centroids, device)

    counts = torch.zeros(K, requires_grad=False).cpu()
    model.eval()
    with torch.no_grad():
        for i_batch, (indice, img_q, sal_q, _, _) in enumerate(dataloader):
            img_q, sal_q = img_q.to(device), sal_q.to(device)
            q, _ = model.model_q(img_q) # Bx dim x H x W
            
            q = nn.functional.normalize(q, dim=1)
            B, dim, H, W = q.shape

            if i_batch == 0:
                logger.info('Centroid size      : {}'.format(list(centroids.shape)))
                logger.info('Batch input size   : {}'.format(list(img_q.shape)))
                logger.info('Batch feature size : {}\n'.format(list(q.shape)))

            # Compute distance and assign label. 
            scores  = compute_negative_euclidean(q, sal_q, centroids, metric_function) 

            # Save labels and count. 
            for idx, _ in enumerate(indice):
                counts += postprocess_label(K, idx, scores)
                

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
            batch_size = q.shape[0]

            probs = classifier(q) #BxdimxHxW
            probs = F.interpolate(probs, label.shape[-2:], mode='bilinear', align_corners=False)
            
            preds = (probs.topk(1, dim=1)[1].squeeze() + 1) * sal 
            preds = preds.view(batch_size, -1).long().cpu().numpy()

            # preds = probs.topk(1, dim=1)[1].view(batch_size, -1).cpu().numpy()
            # label = label.view(batch_size, -1).cpu().numpy()
            
            valid_preds = preds[label != 255]
            # valid_preds = preds[label != 0]
            valid_label = label[label != 255]
            # valid_label = label[label != 0]
            

            histogram += scores(valid_label, valid_preds, args.K_test)
            
            if i_batch%20==0:
                logger.info('{}/{}'.format(i_batch, len(dataloader)))
    
    # Hungarian Matching. 
    m = linear_assignment(histogram.max() - histogram)
    
    # Evaluate. 
    acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum() * 100

    new_hist = np.zeros((args.K_test, args.K_test))
    for idx in range(args.K_test):
        new_hist[m[idx, 1]] = histogram[idx]
    

    # NOTE: Now [new_hist] is re-ordered to 12 thing + 15 stuff classses. 
    res1 = get_result_metrics(new_hist)
    # logger.info('ACC  - All: {:.4f}'.format(res1['overall_precision (pixel accuracy)']))
    # logger.info('mIOU - All: {:.4f}'.format(res1['mean_iou']))

    # # For Table 2 - partitioned evaluation.
    # if args.thing and args.stuff:
    #     res2 = get_result_metrics(new_hist[1:, 1:])
    #     logger.info('ACC  - Thing: {:.4f}'.format(res2['overall_precision (pixel accuracy)']))
    #     logger.info('mIOU - Thing: {:.4f}'.format(res2['mean_iou']))

    #     res3 = get_result_metrics(new_hist[:1, :1])
    #     logger.info('ACC  - Stuff: {:.4f}'.format(res3['overall_precision (pixel accuracy)']))
    #     logger.info('mIOU - Stuff: {:.4f}'.format(res3['mean_iou']))
    
    return acc, res1
