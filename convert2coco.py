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
            # {"supercategory": "passport_code", "id": 4, "name": "passport_code"},
        ],
    }
    # list_class = ['page', 'profile_image', 'van_tay_trai', "van_tay_phai"]
    list_class = {"box1":"page", "box2":"page"}

    classname_to_id = {each['name']: each['id'] for each in coco_json['categories']}
    object_index = 0
    all_sample = glob.glob(os.path.join(subset_path, '*.json'))

    

    for sample_index, sample in tqdm(enumerate(all_sample), total=len(all_sample)):
        raw_anno = json.loads(open(sample, 'r').read())
        # file_path = os.path.join(subset_path, 'images', raw_anno['file_name'])
        file_path = sample.replace("label.json","image.jpg")
        file_name = file_path.split("/")[-1]
        # print(file_path)
        if not os.path.exists(file_path):
            continue
        height, width, _ = cv2.imread(file_path).shape
        id = sample_index
        coco_json['images'].append({
            'file_name': file_name,
            'height': height,
            'width': width,
            'id': id
        })
        for index, each_object in enumerate(raw_anno):
            # x1, y1, x2, y2 = each_object['polygon'][0][0], each_object['polygon'][0][1], \
            #                 each_object['polygon'][2][0], each_object['polygon'][2][1]
            x1 = int(min(np.asarray(each_object['polygon'])[..., 0]))
            y1 = int(min(np.asarray(each_object['polygon'])[..., 1]))
            x2 = int(max(np.asarray(each_object['polygon'])[..., 0]))
            y2 = int(max(np.asarray(each_object['polygon'])[..., 1]))
            if each_object['key'] not in list_class:
                continue
            object_name = list_class[each_object['key']]
            # if 'van_tay' in each_object['key']:
            #     object_name = 'van_tay' 

            # print(each_object)
            
            anno = {
                'id': object_index,
                'segmentation': [[item for sublist in each_object['polygon'] for item in sublist]], # x1,y1, x2,y2, ..
                'area': (x2-x1)*(y2-y1),
                'iscrowd': 0,
                'image_id': id,
                'category_id': classname_to_id[object_name],
                'bbox': [x1, y1, x2-x1, y2-y1]              # left, top, width, height
            }
            
            coco_json['annotations'].append(anno)
            object_index += 1

    return coco_json

if __name__ == "__main__":
    # dateset_basedir = 'data/result_crop_augmented'
    # annotation_path = os.path.join(dateset_basedir, 'annotations')
    # os.makedirs(annotation_path, exist_ok=True)
    # for subset in ['train', 'val']:
    #     print('Converting', subset)
    #     subset_path = os.path.join(dateset_basedir, subset)
    subset_path = "/media/geneous/01D62877FB2A4900/Techainer/Object_detection/MaskRCNN-card-crop-and-rotate/data/ready_data"
    annotation_path = './data/annotations'
    os.makedirs(annotation_path, exist_ok=True)
    coco_json = convert_subset(subset_path)

    with open(os.path.join(annotation_path, 'instances_train_set_vito.json'), 'w', encoding='utf-8') as f:
        json.dump(coco_json, f, ensure_ascii=False, indent=4)



    # old_image_folder = os.path.join(dateset_basedir, subset, 'images')
    # new_image_folder = os.path.join(dateset_basedir, '{}idcard'.format(subset))
    # shutil.move(old_image_folder, new_image_folder)