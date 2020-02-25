python3 predict.py --load log/MaskRCNN-R50C41x-COCO_finetune-docrop_and_rotate/checkpoint \
                   --predict /Users/linus/techainer/vietnamese-identity-card/data/failcase/meh/*

# python3 predict.py --load log/MaskRCNN-R50C41x-COCO_finetune-docrop_and_rotate/checkpoint \
#                    --output-pb log/MaskRCNN-R50C41x-COCO_finetune-docrop_and_rotate/frozen_model.pb