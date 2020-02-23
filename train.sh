python3 train.py --config DATA.BASEDIR=data/result_crop_augmented \
                          MODE_FPN=True \
                          TRAIN.BASE_LR=1e-2 \
                          TRAIN.EVAL_PERIOD=10 \
                          "PREPROC.TRAIN_SHORT_EDGE_SIZE=[512, 512]" \
                          TRAIN.CHECKPOINT_PERIOD=1 \
                          DATA.NUM_WORKERS=2 \
                          EXPERIMENT_NAME=tensorpack_maskRCNN_docrop_and_rotate_overfit_test \
                          --load weights/COCO-MaskRCNN-R50C41x.npz \
                          --logdir log/overfit_test