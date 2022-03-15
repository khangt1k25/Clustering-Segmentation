#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
#
# MoCo implementation based upon https://arxiv.org/abs/1911.05722
# 
# Pixel-wise contrastive loss based upon our paper
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.losses import BalancedCrossEntropyLoss
from modules.fpn import get_model

class ContrastiveModel(nn.Module):
    def __init__(self, args):
        """
        p: configuration dict
        """
        super(ContrastiveModel, self).__init__()

        self.K = 128
        self.m = 0.999
        self.T = 0.5


        # create the model 
        self.model_q = get_model(args)
        self.model_k = get_model(args)
    
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.dim = args.ndim
        
        self.register_buffer("obj_queue", torch.randn(self.dim, self.K))
        self.obj_queue = nn.functional.normalize(self.obj_queue, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


        # balanced cross-entropy loss
        self.bce = BalancedCrossEntropyLoss(size_average=True)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_obj):

        batch_size = keys_obj.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.obj_queue[:, ptr:ptr + batch_size] = keys_obj.T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr


    def mc_forward(self, im_q, sal_q, im_k, sal_k):
        batch_size = im_q.size(0)

        q, q_bg = self.model_q(im_q)         # queries: B x dim x H x W
        q = nn.functional.normalize(q, dim=1)
        q = q.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
        q = torch.reshape(q, [-1, self.dim]) # queries: pixels x dim

        # compute saliency loss
        sal_loss = self.bce(q_bg, sal_q)
   
        with torch.no_grad():
            offset = torch.arange(0, 2 * batch_size, 2).to(sal_q.device)
            sal_q = (sal_q + torch.reshape(offset, [-1, 1, 1]))*sal_q # all bg's to 0
            sal_q = sal_q.view(-1)
            mask_indexes = torch.nonzero((sal_q)).view(-1).squeeze()
            sal_q = torch.index_select(sal_q, index=mask_indexes, dim=0) // 2

        # compute key prototypes
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k, _ = self.model_k(im_k)  # keys: N x C x H x W
            k = nn.functional.normalize(k, dim=1)
           
            # prototypes k
            k = k.reshape(batch_size, self.dim, -1) # B x dim x H.W
            sal_k = sal_k.reshape(batch_size, -1, 1).type(k.dtype) # B x H.W x 1
            prototypes_foreground = torch.bmm(k, sal_k).squeeze() # B x dim
            prototypes = nn.functional.normalize(prototypes_foreground, dim=1)        

        # q: pixels x dim
        # k: pixels x dim
        # prototypes_k: proto x dim
        q = torch.index_select(q, index=mask_indexes, dim=0)
        l_batch = torch.matmul(q, prototypes.t())   # shape: pixels x proto
        negatives = self.obj_queue.clone().detach()     # shape: dim x negatives
        l_mem = torch.matmul(q, negatives)          # shape: pixels x negatives (Memory bank)
        logits = torch.cat([l_batch, l_mem], dim=1) # pixels x (proto + negatives)

        # apply temperature
        logits /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(prototypes) 

        return logits, sal_q, sal_loss
        

    def forward(self, im_q, sal_q, im_k, sal_k, classfier, label):

        batch_size, dim, H, W = im_q.shape
    
        q, q_bg = self.model_q(im_q)         # queries: B x dim x H x W
        q = nn.functional.normalize(q, dim=1)
        
        # compute saliency loss
        sal_loss = self.bce(q_bg, sal_q)

        
        probs = classfier(q) # BxCxHxW
        probs = probs.permute((0, 2, 3, 1))
        probs = torch.reshape(probs, [-1, probs.shape[-1]]) # BHW x C
        label =  label.view(-1, ) #BHW 
        
      

        with torch.no_grad():
            offset = torch.arange(0, 2 * batch_size, 2).to(sal_q.device)
            sal_q = (sal_q + torch.reshape(offset, [-1, 1, 1]))*sal_q # all bg's to 0
            sal_q = sal_q.view(-1)
            mask_indexes = torch.nonzero((sal_q)).view(-1).squeeze()
            sal_q = torch.index_select(sal_q, index=mask_indexes, dim=0) // 2
        
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            k, _ = self.model_k(im_k)  # keys: N x C x H x W
            k = nn.functional.normalize(k, dim=1)
            k = k.reshape(batch_size, self.dim, -1) # B x dim x H.W
            
            sal_k = sal_k.reshape(batch_size, -1, 1).type(q.dtype)
            
            prototypes_obj = torch.bmm(k, sal_k).squeeze() # B x dim
            prototypes_obj = nn.functional.normalize(prototypes_obj, dim=1) 
        
        negatives = self.obj_queue.clone().detach()         # dim x K

        # compute pixel-level loss
        q = q.permute((0, 2, 3, 1))          # B x H x W x dim 
        q = torch.reshape(q, [-1, self.dim]) # BHW x dim
        q = torch.index_select(q, index=mask_indexes, dim=0)  # pixels x dim
        
        l_batch = torch.matmul(q, prototypes_obj.t()) # pixels x B
        l_bank = torch.matmul(q, negatives) # pixels x K
        logits = torch.cat([l_batch, l_bank], dim=1) # pixels x (B+K)

        # compute cluster loss : Not use bg
        probs = torch.index_select(probs, index=mask_indexes, dim=0) # pixels x C
        label = torch.index_select(label, index=mask_indexes, dim=0) 
        opt = False # 1:do not push away other cluster
        if opt:
            label = (label -1).long()
            label = torch.index_select(label, index=mask_indexes, dim=0) # pixels
            probs = probs.gather(1, label.view(-1, 1))
            label = torch.zeros(probs.shape[0]).detach()
        else:
            label = (label -1).long().detach() # pixels
        
                

        mask = torch.ones_like(l_batch).scatter_(1, sal_q.unsqueeze(1), 0.)
        l_batch_negatives = l_batch[mask.bool()].view(l_batch.shape[0], -1)
        
        
        cluster_logits  = torch.cat([probs, l_batch_negatives, l_bank], dim=1) # pixels x (cluster+ B-1+K)
        

        self._dequeue_and_enqueue(prototypes_obj) 


        logits /= self.T
        cluster_logits /= self.T
        
        return logits, sal_q.long(), cluster_logits, label.long(), sal_loss




