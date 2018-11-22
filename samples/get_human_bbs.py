import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import random
import math
import numpy as np
import pickle
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

DETECT_TIMES = []

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def get_bbs_from_image(image, visualise=True):
    # Run detection
    start = time.time()
    results = model.detect([image], verbose=1)
    end = time.time()
    detect_time = end - start
    DETECT_TIMES.append(detect_time)
    print("Detect time:", detect_time)

    # Get results and save person rois in pickle file
    r = results[0]
    if visualise:
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], show_bbox=False)

    person_rois = list([roi for index, roi in enumerate(r['rois'])
                        if class_names[r['class_ids'][index]] == 'person'])

    return person_rois


def dump_bbs_to_pickle(rois, outfile_path):
    with open(outfile_path, 'wb') as outfile:
        pickle.dump(rois, outfile, protocol=2)  # protocol=2 for python2 (HMR uses this)
    print("Rois saved to ", outfile_path)


def main(input_path):

    if os.path.isdir(input_path):
        image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path)
                  if f.endswith('.png') or f.endswith('.jpg')]

        for path in image_paths:
            image = skimage.io.imread(path)
            person_rois = get_bbs_from_image(image, visualise=False)
            bb_pkl_path = os.path.splitext(path)[0] + "_bb_coords.pkl"
            dump_bbs_to_pickle(person_rois, bb_pkl_path)
    else:
        image = skimage.io.imread(input_path)
        person_rois = get_bbs_from_image(image, visualise=False)
        bb_pkl_path = os.path.splitext(input_path)[0] + "_bb_coords.pkl"
        dump_bbs_to_pickle(person_rois, bb_pkl_path)

    print("Average detect time:", np.mean(DETECT_TIMES))

if __name__ == '__main__':
    input_path = sys.argv[1]
    main(input_path)
