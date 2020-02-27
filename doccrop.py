import copy
import glob
import json
import os
import time
import uuid
from collections import namedtuple

import cv2
import imutils
import numpy as np
import tensorflow as tf
from scipy import interpolate
from shapely.geometry import Polygon
from tensorpack.tfutils import get_tf_version_tuple
from tqdm import tqdm

from common import CustomResize, clip_boxes, load_graph, rotate_polygon
from viz import draw_final_outputs_blackwhite

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask', 'polygon'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""


class MaskRCNNDocCrop():
    def __init__(self,
                 model_path='log/MaskRCNN-R50C41x-COCO_finetune-docrop_and_rotate/frozen_model.pb',
                 canvas_size=512,
                 debug=False):
        if not tf.test.is_gpu_available():
            from tensorflow.python.framework import test_util
            assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
                "Inference requires either GPU support or MKL support!"
        self.canvas_size = canvas_size
        self.debug = debug
        self.id_to_class_name = {
            1: 'page',
            2: 'profile_image',
            3: 'van_tay',
            4: 'passport_code'
        }
        self.resizer = CustomResize(
            self.canvas_size, self.canvas_size)
        print('Loading model at', model_path)
        self.graph = load_graph(model_path)
        self.input_tensor = self.graph.get_tensor_by_name('import/image:0')
        self.output_node_name = [
            'output/boxes', 'output/scores', 'output/labels', 'output/masks']
        self.outputs_tensor = [self.graph.get_tensor_by_name(
            'import/{}:0'.format(each_node)) for each_node in self.output_node_name]
        self.config = tf.compat.v1.ConfigProto()
        # self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.compat.v1.Session(config=self.config, graph=self.graph)
        self.predict_crop(np.zeros((200, 200, 3), dtype=np.uint8))
        print('Loaded model!')

    def _scale_box(self, box, scale):
        w_half = (box[2] - box[0]) * 0.5
        h_half = (box[3] - box[1]) * 0.5
        x_c = (box[2] + box[0]) * 0.5
        y_c = (box[3] + box[1]) * 0.5

        w_half *= scale
        h_half *= scale

        scaled_box = np.zeros_like(box)
        scaled_box[0] = x_c - w_half
        scaled_box[2] = x_c + w_half
        scaled_box[1] = y_c - h_half
        scaled_box[3] = y_c + h_half
        return scaled_box

    def _paste_mask(self, box, mask, shape, accurate_paste=True):
        """
        Args:
            box: 4 float
            mask: MxM floats
            shape: h,w
        Returns:
            A uint8 binary image of hxw.
        """
        assert mask.shape[0] == mask.shape[1], mask.shape

        if accurate_paste:
            # This method is accurate but much slower.
            mask = np.pad(mask, [(1, 1), (1, 1)], mode='constant')
            box = self._scale_box(box, float(
                mask.shape[0]) / (mask.shape[0] - 2))

            mask_pixels = np.arange(0.0, mask.shape[0]) + 0.5
            mask_continuous = interpolate.interp2d(
                mask_pixels, mask_pixels, mask, fill_value=0.0)
            h, w = shape
            ys = np.arange(0.0, h) + 0.5
            xs = np.arange(0.0, w) + 0.5
            ys = (ys - box[1]) / (box[3] - box[1]) * mask.shape[0]
            xs = (xs - box[0]) / (box[2] - box[0]) * mask.shape[1]
            # Waste a lot of compute since most indices are out-of-border
            res = mask_continuous(xs, ys)
            return (res >= 0.5).astype('uint8')
        else:
            # This method (inspired by Detectron) is less accurate but fast.

            # int() is floor
            # box fpcoor=0.0 -> intcoor=0.0
            x0, y0 = list(map(int, box[:2] + 0.5))
            # box fpcoor=h -> intcoor=h-1, inclusive
            x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
            x1 = max(x0, x1)    # require at least 1x1
            y1 = max(y0, y1)

            w = x1 + 1 - x0
            h = y1 + 1 - y0

            # rounding errors could happen here, because masks were not originally computed for this shape.
            # but it's hard to do better, because the network does not know the "original" scale
            mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
            ret = np.zeros(shape, dtype='uint8')
            ret[y0:y1 + 1, x0:x1 + 1] = mask
            return ret

    def predict_crop(self, img, debug_id=None):
        start_time = time.time()
        orig_shape = img.shape[:2]
        resized_img = self.resizer.augment(img)
        scale = np.sqrt(
            resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
        boxes, probs, labels, *masks = self.sess.run(self.outputs_tensor, feed_dict={
            self.input_tensor: resized_img})

        # Some slow numpy postprocessing:
        boxes = boxes / scale
        # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
        boxes = clip_boxes(boxes, orig_shape)
        if masks:
            full_masks = [self._paste_mask(box, mask, orig_shape)
                          for box, mask in zip(boxes, masks[0])]
            masks = full_masks
        else:
            # fill with none
            masks = [None] * len(boxes)

        polygons = []
        # Estimate polygon based on the mask right here
        for mask in masks:
            temp_mask = np.expand_dims(mask, axis=-1)*255
            cnts = cv2.findContours(
                temp_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnt = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            estimated_polygon = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            polygons.append(estimated_polygon)
            # temp_mask = cv2.cvtColor(temp_mask, cv2.COLOR_GRAY2BGR)
            # viz_img = cv2.polylines(temp_mask, [estimated_polygon], isClosed=True, color=(255, 0, 255), thickness=10)
            # cv2.imwrite('mask.png', viz_img)
            # import ipdb; ipdb.set_trace()

        results = [DetectionResult(*args)
                   for args in zip(boxes, probs, labels.tolist(), masks, polygons)]

        if self.debug:
            print('Crop tooks {} secs.'.format(time.time()-start_time))
            debug_id = uuid.uuid4() if debug_id is None else debug_id
            debug_path = os.path.join('./debugs/', '{}.png'.format(debug_id))
            # os.makedirs(debug_path, exist_ok=True)
            final = draw_final_outputs_blackwhite(img, results)
            cv2.imwrite(debug_path, final)
            # cv2.imwrite(os.path.join(debug_path, 'crop.png'), final)
        return results

    def rotate_anno(self, all_object, angle, raw_img_shape, before_shape=None):
        new_object = []
        # Create full page mask
        old_shape = before_shape if before_shape is not None else raw_img_shape[::-1]
        full_page = [(0, 0), (old_shape[1], 0),
                     (old_shape[1], old_shape[0]), (0, old_shape[0])]
        rotated_full_page = rotate_polygon(full_page, angle, raw_img_shape)
        # Calculate offset of rotation
        top_left_x = min([each[0] for each in rotated_full_page])
        top_left_y = min([each[1] for each in rotated_full_page])
        for obj in all_object:
            rotated_obj = obj
            org_polygon = [(each[0][0], each[0][1]) for each in obj.polygon]
            rotated_polygon = rotate_polygon(
                org_polygon, angle, raw_img_shape, top_left_x, top_left_y)
            rotated_obj._replace(polygon=rotated_polygon)
            new_object.append(rotated_obj)
        return new_object

    def get_overlap_object(self, page, page_index, cropped_result_raw):
        cropped_result = copy.deepcopy(cropped_result_raw)
        page_polygon = Polygon([(each[0][0], each[0][1]) for each in page.polygon])
        if not page_polygon.is_valid:
            page_polygon = page_polygon.buffer(0)
        del cropped_result[page_index]
        overlaped_object = []
        for each_object in cropped_result:
            obj_polygon = Polygon([(each[0][0], each[0][1]) for each in each_object.polygon])
            if not obj_polygon.is_valid:
                obj_polygon = obj_polygon.buffer(0)
            intersec_percentage = obj_polygon.intersection(
                page_polygon).area/obj_polygon.area
            if intersec_percentage >= 0.85:
                overlaped_object.append(each_object)
        return overlaped_object

    def keep_only_biggest(self, all_object):
        group_dict = {k.class_id:[] for k in all_object}
        filtered_object = []
        for each in all_object:
            group_dict[each.class_id].append(each)
        for each_group, all_members in group_dict.items():
            area_list = []
            for each_member in all_members:
                each_polygon = Polygon(
                    [(each[0][0], each[0][1]) for each in each_member.polygon])
                if not each_polygon.is_valid:
                    each_polygon = each_polygon.buffer(0)
                area_list.append(each_polygon.area)
            filtered_object.append(all_members[np.argmax(area_list)])
        return filtered_object

    def refine_object_location(self, page_bbox, other_objects):
        x, y = page_bbox[0], page_bbox[1]
        new_objects = []
        for index, each in enumerate(other_objects):
            refined_object = each
            old_bb = refined_object.box
            new_bb = np.array([old_bb[0]-x, old_bb[1]-y, old_bb[2]-x, old_bb[3]-y], dtype=np.int32)
            old_polygon = np.squeeze(refined_object.polygon)
            new_polygon = np.array([[e[0]-x, e[1]-y] for e in old_polygon], dtype=np.int32)
            new_polygon = np.expand_dims(new_polygon, axis=1)
            refined_object = refined_object._replace(box=new_bb, polygon=new_polygon)
            new_objects.append(refined_object)
        return new_objects

    def find_object_by_name(self, all_object, field_name):
        for each_object in all_object:
            if self.id_to_class_name[each_object.class_id] == field_name:
                return each_object
        return None

    def big_rotate_without_anchor(self, page, all_object):
        pass

    def big_rotate_with_anchor(self, page, all_object, anchor_field):
        if anchor_field == 'profile_image' or 'van_tay':
            anchor_object = self.find_object_by_name(all_object, anchor_field)

        return

    def crop_and_rotate(self, image, debug_id=None):
        cropped_result = self.predict_crop(image, debug_id)
        all_pages_result = [(index, each) for index, each in enumerate(cropped_result) if each.class_id == 1]
        results = []
        for page_index, each_page in all_pages_result:
            # Find what other object are accosiated with this page
            other_objects = self.get_overlap_object(each_page, page_index, cropped_result)
            
            # Crop the page in raw img
            page_bbox = [int(each) for each in each_page.box]
            cropped_page = image[page_bbox[1]:page_bbox[3], page_bbox[0]:page_bbox[2]]
            # And then refine the page polygon
            each_page = self.refine_object_location(page_bbox, [each_page])[0]

            if other_objects:
                # Then clear the duplicate by one using the biggest
                other_objects = self.keep_only_biggest(other_objects)
                # And refine the accosiated location right now
                other_objects = self.refine_object_location(page_bbox, other_objects)

            # Then do fine rotation for the whole group first
            # Now we estimate the rotation angle of the page
            angle = cv2.minAreaRect(each_page.polygon)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # Then rotate the whole bounding box including the page and the anno associated with that page too
            before_rotate_shape = cropped_page.shape[:-1]
            cropped_page = imutils.rotate_bound(
                cropped_page, angle=angle, cval=(255, 255, 255))
            after_rotate_shape = cropped_page.shape[:-1]

            # After this gota crop the page and refine all polygon again
            each_page = self.rotate_anno([each_page], angle, before_rotate_shape, after_rotate_shape)[0]
            page_polygon = [(each[0][0], each[0][1]) for each in each_page.polygon]
            all_X = [each[0] for each in page_polygon]
            all_Y = [each[1] for each in page_polygon]
            page_bbox = [min(all_X), min(all_Y), max(all_X), max(all_Y)]
            cropped_page = cropped_page[page_bbox[1]:page_bbox[3], page_bbox[0]:page_bbox[2]]
            each_page = self.refine_object_location(page_bbox, [each_page])[0]
            if other_objects:
                other_objects = self.rotate_anno(other_objects, angle, before_rotate_shape, after_rotate_shape)
                other_objects = self.refine_object_location(page_bbox, other_objects)
            
            # viz_img = cv2.polylines(cropped_page, [x.polygon for x in other_objects], isClosed=True, color=(0, 255, 255), thickness=2)
            # cv2.imwrite('mask.png', viz_img)
            # import ipdb; ipdb.set_trace()
            
            # # Now we do big rotation like 90 or 180 :P
            # each_page, other_objects = self.big_rotate_without_anchor(each_page, other_objects)

            # # Then use some anchor point to correct upsidedown cases
            # other_object_name = [self.id_to_class_name(each.class_id) for each in other_objects]
            # # If fingerprint, face and mrz all appear, use the one with highest confidents
            # # Priority profile image and fingerprint first
            # most_conf = max(other_objects, key=lambda x: x.score)
            # anchor_field = self.id_to_class_name[most_conf.class_id]
            # do_it = False
            # if 'profile_image' in other_object_name and anchor_field != 'profile_image':
            #     profile_image_object = other_objects[other_object_name.index('profile_image')]
            #     if abs(profile_image_object.score-most_conf.score) <= 0.05:
            #         anchor_field = 'profile_image'
            #         do_it = True
            # if not do_it and 'van_tay' in other_object_name and anchor_field != 'van_tay':
            #     profile_image_object = other_objects[other_object_name.index('van_tay')]
            #     if abs(profile_image_object.score-most_conf.score) <= 0.05:
            #         anchor_field = 'van_tay'
            # each_page, other_objects = self.big_rotate_with_anchor(each_page, other_objects, anchor_field)


if __name__ == "__main__":
    model = MaskRCNNDocCrop(debug=True)
    all_samples = glob.glob('/Users/linus/Downloads/passport 2/passport_other_countries/others/*')
    all_samples = ['/Users/linus/techainer/vietnamese-identity-card/data/failcase/meh/IMG_0817.JPG']
    for sample in tqdm(all_samples):
        print(sample)
        img = cv2.imread(sample)
        results = model.crop_and_rotate(img, debug_id=os.path.basename(sample))
