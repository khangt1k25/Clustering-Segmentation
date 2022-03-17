K_train=20
K_test=20
bsize_cluster=32
bsize_train=32
bsize_test=32
num_epoch=60
KM_INIT_TRAIN=64 
KM_NUM_TRAIN=3
KM_INIT_TEST=16 
KM_NUM_TEST=3
KM_ITER=20
SEED=2
LR=4e-3
reducer=0
coeff=1e-1

mkdir -p results/train/${SEED}

python train_mdc_pascal.py \
--data_root '/content/drive/MyDrive/UCS_local/PASCAL_VOC' \
--save_root results/train/${SEED} \
--backbone 'resnet50' \
--pretrain \
--pretraining 'imagenet_classification' \
--moco_state_dict '/content/drive/MyDrive/UCS_local/moco_v2_800ep_pretrain.pth.tar' \
--lr ${LR} \
--seed ${SEED} \
--K_train ${K_train} --K_test ${K_test} \
--batch_size_cluster ${bsize_cluster}  \
--batch_size_train ${bsize_train} \
--batch_size_test ${bsize_test} \
--num_init_batches_train ${KM_INIT_TRAIN} \
--num_batches_train ${KM_NUM_TRAIN} \
--num_init_batches_test ${KM_INIT_TEST} \
--num_batches_test ${KM_NUM_TEST} \
--kmeans_n_iter ${KM_ITER} \
--num_epoch ${num_epoch} \
--res 224 \
--augment --jitter --blur --grey --equiv --h_flip --v_flip \
--eval_interval 10 \
--repeats 1 \
--reducer ${reducer} \
--coeff ${coeff} \

