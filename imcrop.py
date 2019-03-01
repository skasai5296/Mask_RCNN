import os
import sys
import random
import math
import time
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from PIL import Image

ROOT_DIR = os.getcwd()

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_weights(COCO_MODEL_PATH, by_name=True)

def normalize_image(array):
    dst = array / 255
    return dst

# Load a random image from the images folder
IMAGE_DIR = os.path.join(ROOT_DIR, "../../local/CelebA/img_align_celeba")
file_names = next(os.walk(IMAGE_DIR))[2]

savepth = "../../local/CelebA/cropped"
SAVE_DIR = os.path.join(ROOT_DIR, savepth)
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

for i, f in enumerate(file_names):
    if os.path.exists(os.path.join(SAVE_DIR, f)):
        continue
    image = skimage.io.imread(os.path.join(IMAGE_DIR, f))

    print("processing {}".format(f), flush=True)
    # Run detection
    attempts = 0
    elapsed = time.time()
    while attempts < 3:
        try:
            results = model.detect([image], verbose=1)
            break
        except:
            attempts += 1
            print("failed to detect, {}th attempt".format(attempts), flush=True)


    r = results[0]

    # get highest scored index (likely to be 'person') and create mask
    try:
        idx = np.where(r['class_ids'] == 1)[0]
        mask = r['masks'][:, :, np.argmax(r['scores'])]
        mask = np.stack([mask] * 3, -1)
        maski = np.invert(mask).astype(int)
        im_crop = normalize_image(image * mask) + maski
    except ValueError:
        im_crop = image

    skimage.io.imsave(os.path.join(SAVE_DIR, f), im_crop)
    print("{}th loop, took {}s".format(i+1, time.time() - elapsed), flush=True)
print("finished cropping {} images".format(len(file_names)))



# additional cropping of broken files
brokenfiles = []
cnt = 0
for name in file_names:
    try:
        im = Image.open(os.path.join('../../local/CelebA/cropped/', name))
    except:
        cnt += 1
        brokenfiles.append(name)
print('brokencnt = {}'.format(cnt), flush=True)


for i, name in enumerate(brokenfiles):
    image = skimage.io.imread(os.path.join(IMAGE_DIR, name))

    print("processing broken {}".format(name), flush=True)
    # Run detection
    attempts = 0
    elapsed = time.time()
    while attempts < 3:
        try:
            results = model.detect([image], verbose=1)
            break
        except:
            attempts += 1
            print("failed to detect, {}th attempt".format(attempts), flush=True)

    r = results[0]

    # get highest scored index (likely to be 'person') and create mask
    try:
        idx = np.where(r['class_ids'] == 1)[0]
        mask = r['masks'][:, :, np.argmax(r['scores'])]
        mask = np.stack([mask] * 3, -1)
        maski = np.invert(mask).astype(int)
        im_crop = normalize_image(image * mask) + maski
    except ValueError:
        im_crop = image

    skimage.io.imsave(os.path.join(SAVE_DIR, name), im_crop)
    print("{}/{} done , took {}s".format(i+1, cnt, time.time() - elapsed), flush=True)
print("finished cropping {} images, no more broken files".format(cnt))


"""
class InferenceConfig2(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 32

def normalize_image(array):
    dst = array / 255
    return dst

config = InferenceConfig2()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

IMAGE_DIR = os.path.join(ROOT_DIR, "../../local/CelebA/img_align_celeba")
file_names = next(os.walk(IMAGE_DIR))[2]
batch = []
savepth = '../../local/CelebA/cropped'
SAVEDIR = os.path.join(ROOT_DIR, savepth)
file_names = [impth for impth in file_names if not os.path.exists(os.path.join(SAVEDIR, impth))]

def image_generator(bs, filelist):
    for i in range(len(filelist) // bs):
        if (i+1) * bs < len(filelist):
            endnum = (i+1) * bs
        else:
            break
    batchpaths = filelist[i*bs:endnum]
    imbatch = [skimage.io.imread(os.path.join(IMAGE_DIR, impath)) for impath in batchpaths]
    yield batchpaths, imbatch

bs = config.BATCH_SIZE
loader = image_generator(bs, file_names)
cnt = 0
for batchpaths, imbatch in loader:
    elapsed = time.time()
    results = model.detect(imbatch, verbose=0)
    cropped = []
    for i, r in enumerate(results):
        try:
            idx = np.where(r['class_ids'] == 1)[0]
            mask = r['masks'][:, :, np.argmax(r['scores'])]
            mask = np.stack([mask] * 3, -1)
            maski = np.invert(mask).astype(int)
            ans = normalize_image(imbatch[i] * mask) + maski
        except ValueError:
            ans = imbatch[i]
        pth = os.path.join(SAVEDIR, batchpaths[i])
        skimage.io.imsave(pth, ans)
    cnt += bs
    print("{}th image saved, took {}s".format(cnt, (time.time() - elapsed) / bs), flush=True)

"""
