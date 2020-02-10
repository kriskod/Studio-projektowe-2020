import argparse
import logging
import sys
import shutil
import time
import scipy
import os
import matplotlib.pyplot as plt
from scipy.io import savemat
import numpy
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !

    iterator = 0;
    pictureDir = './human_dataset/'
    goodPictureDir = './goodPics/'

    xArr = []
    yArr = []
    zArr = []

    for i in range(17):
        xArr.append([])
        yArr.append([])
        zArr.append([])

    for filename in os.listdir(pictureDir):
        imagePath = pictureDir + '/' + filename

        image = common.read_imgfile(imagePath, None, None)
        if image is None:
            logger.error('Image can not be read, path=%s' % imagePath)
            sys.exit(-1)

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        image_h, image_w = image.shape[:2]

        max = 0
        max_idx = 0
        for num, human in enumerate(humans, start=0):
            if(human.score> max):
                max_idx = num
                max = human.score

        coordinates = {}


        if(humans.__len__()!=0):
            human = humans[max_idx]
            iter = 0
            if(human.body_parts.__len__() == 17):
                print(filename)
                centers = {}


                for i in range(common.CocoPart.Background.value):
                    if i not in human.body_parts.keys():
                        continue

                    body_part = human.body_parts[i]
                    center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                    centers[i] = center

                    x = float(center[0])
                    y = float(center[1])
                    z = float(1)

                    xArr[iter].append(x)
                    yArr[iter].append(y)
                    zArr[iter].append(z)
                    iter += 1

                shutil.copyfile(imagePath, goodPictureDir + str(iterator).zfill(4) + '.jpg')
                iterator += 1

    mat = [xArr, yArr, zArr]
    newJoints = np.array(mat)

    savemat('result.mat', {"joints": newJoints})
    print()


