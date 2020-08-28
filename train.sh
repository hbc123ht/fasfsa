python3 train.py --config DATA.BASEDIR=./data \
                          MODE_FPN=True \
                          TRAIN.BASE_LR=1e-2 \
                          TRAIN.EVAL_PERIOD=5 \
                          "PREPROC.TRAIN_SHORT_EDGE_SIZE=[512, 512]" \
                          TRAIN.CHECKPOINT_PERIOD=5 \
                          DATA.NUM_WORKERS=2 \
                          EXPERIMENT_NAME=MaskRCNN-R50C41x-COCO_finetune_seperate_fingerprint-docrop_and_rotate \
                          --load weights/COCO-MaskRCNN-R50C41x.npz \
                          --logdir log/MaskRCNN-R50C41x-COCO_finetune_seperate_fingerprint-docrop_and_rotate