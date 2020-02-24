import cv2
import os
import glob
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import shutil


def convert_subset(subset_path):
    coco_json = {
        "info": {
            "description": '{} dataset'.format(subset_path),
            "url": "http://techainer.com",
            "version": "1.0",
            "year": 2020,
            "contributor": "linus",
            "date_created": datetime.today().strftime('%Y-%m-%d')
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "page", "id": 1, "name": "page"},
            {"supercategory": "profile_image", "id": 2, "name": "profile_image"},
            {"supercategory": "van_tay", "id": 3, "name": "van_tay"},
            {"supercategory": "passport_code", "id": 4, "name": "passport_code"},
        ],
    }

    classname_to_id = {each['name']: each['id'] for each in coco_json['categories']}

    all_sample = glob.glob(os.path.join(subset_path, 'annos', '*.json'))
    for sample in tqdm(all_sample):
        raw_anno = json.loads(open(sample, 'r').read())
        file_path = os.path.join(subset_path, 'images', raw_anno['file_name'])
        if not os.path.exists(file_path):
            continue
        height, width, _ = cv2.imread(file_path).shape
        id = raw_anno['file_name'].replace('.png', '')
        coco_json['images'].append({
            'file_name': raw_anno['file_name'],
            'height': height,
            'width': width,
            'id': id
        })
        for each_object in raw_anno['regions']:
            x1, y1, x2, y2 = each_object['bbox']
            anno = {
                'segmentation': [[[item for sublist in each_object['segmentation'] for item in sublist]]],
                'area': 0,
                'iscrowd': 0,
                'image_id': id,
                'category_id': classname_to_id[each_object['class']],
                'bbox': [x1, y1, x2-x1, y2-y1]
            }
            coco_json['annotations'].append(anno)
    return coco_json

if __name__ == "__main__":
    dateset_basedir = 'data/result_crop_augmented'
    annotation_path = os.path.join(dateset_basedir, 'annotations')
    os.makedirs(annotation_path, exist_ok=True)
    for subset in ['train', 'val']:
        print('Converting', subset)
        subset_path = os.path.join(dateset_basedir, subset)
        coco_json = convert_subset(subset_path)
        with open(os.path.join(annotation_path, 'instances_{}idcard.json'.format(subset)), 'w', encoding='utf-8') as f:
            json.dump(coco_json, f, ensure_ascii=False, indent=4)
        old_image_folder = os.path.join(dateset_basedir, subset, 'images')
        new_image_folder = os.path.join(dateset_basedir, '{}idcard'.format(subset))
        shutil.move(old_image_folder, new_image_folder)