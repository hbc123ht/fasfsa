import time
import glob
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from collections import namedtuple

import cv2
import numpy as np
import tensorflow as tf
from scipy import interpolate
from tensorpack.tfutils import get_tf_version_tuple
from tqdm import tqdm

from common import CustomResize, clip_boxes, load_graph
from viz import draw_final_outputs_blackwhite


DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""


class MaskRCNNDocCrop():
    def __init__(self,
                 model_path='log/MaskRCNN-R50C41x-COCO_finetune-docrop_and_rotate/frozen_model.pb',
                 canvas_size=512):
        # if not tf.test.is_gpu_available():
        #     from tensorflow.python.framework import test_util
        #     assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
        #         "Inference requires either GPU support or MKL support!"
        self.canvas_size = canvas_size
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

    def predict(self, img):
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

        results = [DetectionResult(*args)
                   for args in zip(boxes, probs, labels.tolist(), masks)]
        return results

if __name__ == "__main__":
    model = MaskRCNNDocCrop()
    os.makedirs('./debugs', exist_ok=True)
    all_samples = glob.glob('/Users/linus/techainer/vietnamese-identity-card/data/failcase/meh/*')
    for sample in tqdm(all_samples):
        print(sample)
        img = cv2.imread(sample)
        start_time = time.time()
        results = model.predict(img)
        print('Tooks {} secs.'.format(time.time()-start_time))
        final = draw_final_outputs_blackwhite(img, results)
        cv2.imwrite('./debugs/{}'.format(os.path.basename(sample)), final)
