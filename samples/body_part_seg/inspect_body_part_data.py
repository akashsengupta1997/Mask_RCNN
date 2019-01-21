import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.body_part_seg import body_part_seg

config = body_part_seg.BodyPartsConfig()
UPS31_DIR = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31/images"

# Load dataset
# Get the dataset from the releases page
# https://github.com/matterport/Mask_RCNN/releases
dataset = body_part_seg.BodyPartsDataset()
dataset.load_UPS31(UPS31_DIR, "train")

# Must call before using the dataset
dataset.prepare()

# --- TEST MASKS AND IMAGES ---
# print("Image Count: {}".format(len(dataset.image_ids)))
# print("Class Count: {}".format(dataset.num_classes))
# for i, info in enumerate(dataset.class_info):
#     print("{:3}. {:50}".format(i, info['name']))
#
# # Load and display random samples
# image_ids = np.random.choice(dataset.image_ids, 4)
# for image_id in image_ids:
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

# # Load random image and mask.
# image_id = random.choice(dataset.image_ids)
# image = dataset.load_image(image_id)
# mask, class_ids = dataset.load_mask(image_id)
# # Compute Bounding box
# bbox = utils.extract_bboxes(mask)
#
# # Display image and additional stats
# print("image_id ", image_id, dataset.image_reference(image_id))
# log("image", image)
# log("mask", mask)
# log("class_ids", class_ids)
# log("bbox", bbox)
# # Display image and instances
# visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

# # --- TEST BOUNDING BOXES ---
# # Load random image and mask.
# image_id = np.random.choice(dataset.image_ids, 1)[0]
# image = dataset.load_image(image_id)
# mask, class_ids = dataset.load_mask(image_id)
# original_shape = image.shape
# # Resize
# image, window, scale, padding, _ = utils.resize_image(
#     image,
#     min_dim=config.IMAGE_MIN_DIM,
#     max_dim=config.IMAGE_MAX_DIM,
#     mode=config.IMAGE_RESIZE_MODE)
# mask = utils.resize_mask(mask, scale, padding)
# # Compute Bounding box
# bbox = utils.extract_bboxes(mask)
#
# # Display image and additional stats
# print("image_id: ", image_id, dataset.image_reference(image_id))
# print("Original shape: ", original_shape)
# log("image", image)
# log("mask", mask)
# log("class_ids", class_ids)
# log("bbox", bbox)
# # Display image and instances
# visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

#
# # --- TEST RESIZING ---
# image_id = np.random.choice(dataset.image_ids, 1)[0]
# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#     dataset, config, image_id, use_mini_mask=False)
# log("image", image)
# log("image_meta", image_meta)
# log("class_ids", class_ids)
# log("bbox", bbox)
# log("mask", mask)
# display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])
# visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
#
# # Add augmentation and mask resizing.
# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#     dataset, config, image_id, augment=True, use_mini_mask=True)
# log("mask", mask)
# display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])
# mask = utils.expand_mask(bbox, mask, image.shape)
# visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


# --- TEST ANCHORS ---

