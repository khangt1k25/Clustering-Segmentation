K_train=100
K_test=20
bsize=32
num_epoch=30
KM_INIT=32 # need
KM_NUM=32 #need
KM_ITER=20
SEED=5
LR=1e-4

mkdir -p results/mdc/train/${SEED}

python train_mdc_pascal.py \
--data_root '/content/drive/MyDrive/UCS_local/PASCAL_VOC' \
--save_root results/pascal/train/${SEED} \
--pretrain \
--repeats 1 \
--lr ${LR} \
--seed ${SEED} \
--num_init_batches ${KM_INIT} \
--num_batches ${KM_NUM} \
--kmeans_n_iter ${KM_ITER} \
--K_train ${K_train} --K_test ${K_test} \
--batch_size_cluster ${bsize}  \
--num_epoch ${num_epoch} \
--res 224 \
--augment --jitter --blur --grey --equiv --random_crop --h_flip --v_flip
