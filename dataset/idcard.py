import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry
import glob
import cv2

__all__ = ["register_idcard"]

class IDCardDataset(DatasetSplit):
    def __init__(self, data_path, split):
        assert split in ["train", "val"]
        self.data_path = os.path.join(data_path, split)
        assert os.path.isdir(self.data_path) and os.path.exists(self.data_path), "Dataset path {} doesn't exist or is not a directory"
        self.all_samples = glob.glob(os.path.join(self.data_path, 'annos', '*.json'))
        assert len(self.all_samples) != 0, "The dataset is empty"
        self.images_path = os.path.join(self.data_path, 'images')
        self.class_name_to_id = {
            'page': 1,
            'profile_image': 2,
            'van_tay': 3,
            'passport_code': 4
        }
        self.class_names = list(self.class_name_to_id.keys())

    def training_roidbs(self):
        ret = []
        for sample_path in self.all_samples:
            raw_anno = json.loads(open(sample_path, 'r').read())
            raw_image_path = os.path.join(self.images_path, raw_anno['file_name'])
            if not os.path.exists(raw_image_path):
                print('{} is missing images at {}. Skiped!'.format(sample_path, raw_image_path))
                continue
            roidb = {
                "file_name": raw_image_path,
                "image_id": raw_anno['file_name']
                }
            boxes = []
            segs = []
            classes = []
            for each_object in raw_anno['regions']:
                boxes.append(each_object['bbox'])
                poly = np.array(each_object['segmentation'])
                segs.append([poly])
                classes.append(self.class_name_to_id[each_object['class']])
            roidb["boxes"] = np.asarray(boxes, dtype=np.float32)
            roidb["segmentation"] = segs
            N = len(raw_anno['regions'])
            roidb["class"] = np.asarray(classes, dtype=np.int32)
            roidb["is_crowd"] = np.zeros((N, ), dtype=np.int8)
            ret.append(roidb)
        return ret

    def inference_roidbs(self):
        return self.training_roidbs()

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    def print_coco_metrics(self, results):
        """
        Args:
            results(list[dict]): results in coco format
        Returns:
            dict: the evaluation metrics
        """
        from pycocotools.cocoeval import COCOeval
        ret = {}
        # results will be modified by loadRes
        has_mask = "segmentation" in results[0]

        cocoDt = self.coco.loadRes(results)
        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        fields = ['IoU=0.5:0.95', 'IoU=0.5',
                  'IoU=0.75', 'small', 'medium', 'large']
        for k in range(6):
            ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

        if len(results) > 0 and has_mask:
            cocoEval = COCOeval(self.coco, cocoDt, 'segm')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            for k in range(6):
                ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
        return ret


    def eval_inference_results(self, results, output=None):
        # continuous_id_to_COCO_id = {v: k for k,
        #                             v in self.COCO_id_to_category_id.items()}
        for res in results:
            # convert to COCO's incontinuous category id
            # if res['category_id'] in continuous_id_to_COCO_id:
            #     res['category_id'] = continuous_id_to_COCO_id[res['category_id']]
            # COCO expects results in xywh format
            box = res['bbox']
            box[2] -= box[0]
            box[3] -= box[1]
            res['bbox'] = [round(float(x), 3) for x in box]

        if output is not None:
            with open(output, 'w') as f:
                json.dump(results, f)
        if len(results):
            # sometimes may crash if the results are empty?
            return self.print_coco_metrics(results)
        else:
            return {}

def register_idcard(basedir):
    for split in ["train", "val"]:
        name = "idcard_" + split
        DatasetRegistry.register(name, lambda x=split: IDCardDataset(basedir, x))
        DatasetRegistry.register_metadata(
            name, "class_names", ["BG", "page", "profile_image", "van_tay", "passport_code"])


if __name__ == '__main__':
    data_path = 'data/result_crop_augmented'
    register_idcard(data_path)
    roidbs = IDCardDataset(data_path, "train").training_roidbs()
    print("#images:", len(roidbs))

    from viz import draw_annotation
    from tensorpack.utils.viz import interactive_imshow as imshow
    import cv2
    for r in roidbs:
        im = cv2.imread(r["file_name"])
        vis = draw_annotation(im, r["boxes"], r["class"], r["segmentation"])
        imshow(vis)
