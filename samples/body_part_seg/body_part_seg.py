import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class BodyPartsConfig(Config):
    """Configuration for training on UP-S31 dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "body-parts"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 6

    # Number of classes (including background)
    NUM_CLASSES = 31 + 1  # Background + 31 body-parts

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500

    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.5

    # Number of validation steps to run at the end of every training epoch.
    VALIDATION_STEPS = 50

    BACKBONE = 'resnet50'

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 300
    IMAGE_MAX_DIM = 512


############################################################
#  Dataset
############################################################

class BodyPartsDataset(utils.Dataset):

    def load_UPS31(self, dataset_dir, subset):
        """Load a subset of the UP-S31 dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        num_classes = 31  # 31 classes not including background
        for i in range(num_classes):
            self.add_class("up-s31", i, str(i+1))

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        for image_path in sorted(os.listdir(dataset_dir)):
            if image_path.endswith('.png'):
                image_id = int(image_path[:5])
                self.add_image(
                    "up-s31",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, image_path))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance. In the case of UP-S31, instance count = number of classes
        class_ids: a 1D array of class IDs of the instance masks.
        """
        num_classes = 31  # excluding body parts
        image_path = self.image_info[image_id]['path']
        s31_folder = os.path.dirname(os.path.dirname(os.path.dirname(image_path)))
        mask_dir = os.path.join(s31_folder, 'masks', 'train')
        mask_path = os.path.join(mask_dir, str(image_id).zfill(5) + '_ann.png')
        mask_image = skimage.io.imread(mask_path, as_gray=True)
        # from matplotlib import pyplot as plt
        # plt.imshow(mask_image, cmap='gray')
        # plt.show()

        # For UP-S31, 1 instance per class => number of instances = number of classes
        mask = np.zeros((mask_image.shape[0], mask_image.shape[1], num_classes))
        class_ids = np.arange(num_classes, dtype=np.int32)
        for pixel_class in range(1, num_classes+1): # body part labels = 1-31
            indexes = list(zip(*np.where(mask_image == pixel_class)))
            for index in indexes:
                mask[index[0], index[1], pixel_class-1] = 1.0

        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "up-s31":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Training
############################################################

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BodyPartsDataset()
    dataset_train.load_UPS31(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BodyPartsDataset()
    dataset_val.load_UPS31(args.dataset, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                augmentation=augmentation,
                layers='all')


############################################################
#  Predicting
############################################################

def construct_seg_image(shape, masks, class_ids):
    # Construct HxW seg image with each pixel labelled by class
    seg_image = np.zeros(shape)
    for instance in range(len(class_ids)):
        print("Instance", instance)
        class_id = class_ids[instance]
        print("Class", class_id)
        mask = masks[:, :, instance]
        indexes = list(zip(*np.where(mask)))
        for index in indexes:
            seg_image[index] = class_id

    return seg_image


def predict_bodyparts(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        if os.path.isdir(image_path):
            print("Running on {}".format(args.image))
            for image_file in sorted(os.listdir(image_path)):
                # Read image
                image = skimage.io.imread(os.path.join(image_path, image_file))
                # Detect objects
                r = model.detect([image], verbose=1)[0]
                masks = r["masks"]
                class_ids = r["class_ids"]
                seg_image = construct_seg_image(image.shape[:2], masks, class_ids)
                # Save output
                outfile_name = image_file + "_predict_up-s31.png"
                skimage.io.imsave(outfile_name, seg_image)

        elif os.path.isfile(image_path):
            # Run model detection
            print("Running on {}".format(args.image))
            # Read image
            image = skimage.io.imread(args.image)
            # Detect objects
            r = model.detect([image], verbose=1)[0]
            masks = r["masks"]
            class_ids = r["class_ids"]
            seg_image = construct_seg_image(image.shape[:2], masks, class_ids)
            # from matplotlib import pyplot as plt
            # plt.imshow(seg_image * 8)
            # plt.show()
            # Save output
            file_name = os.path.basename(image_path) + "_predict_up-s31.png"
            skimage.io.imsave(file_name, seg_image)
        else:
            print('Invalid path.')
            return None

    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "bodyparts-ups31_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                masks = r["masks"]
                class_ids = r["class_ids"]
                seg_image = construct_seg_image(image.shape[:2], masks, class_ids)

                # Add image to video writer
                vwriter.write(seg_image)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training/Predicting
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect body parts.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'predict'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/train/dataset/",
                        help='Directory of the training dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to detect body parts on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "predict":
        assert args.image or args.video,\
               "Provide --image or --video to predict bodyparts"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BodyPartsConfig()
    else:
        class InferenceConfig(BodyPartsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "predict":
        predict_bodyparts(model, image_path=args.image,
                          video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'predict'".format(args.command))
