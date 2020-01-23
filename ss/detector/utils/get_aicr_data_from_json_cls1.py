import numpy as np
import os
from bisect import bisect
# from xml.etree import ElementTree
import xml.etree.cElementTree as ElementTree
import json
# from lxml.etree import Element as ElementTree
import random


class Json_preprocessor(object):

    def __init__(self, data_path, image_path=None, num_classes=4, ratio=1.0):
        self.path_prefix = data_path
        self.num_classes = num_classes
        self.data = dict()
        self.path_image = image_path
        self._preprocess_json(ratio=ratio)

    def _get_key(self, item):
        return item[0]

    def _preprocess_json(self, ratio=1.0):
        filenames = os.listdir(self.path_prefix)
        k = int(len(filenames) * ratio)
        filenames = random.sample(filenames, k)
        images = []
        exts = []
        if self.path_image is not None:
            tmp = os.listdir(self.path_image)
            tmp = [os.path.splitext(filenm) for filenm in tmp]
            tmp = sorted(tmp, key=self._get_key)
            images, exts = zip(*tmp)

        print('total : ' + str(len(filenames)))
        count = 0
        divisor = len(filenames) // 100
        for filename in filenames:

            with open(self.path_prefix + filename, 'r') as fp:
                jsonroot = json.load(fp)

            boxes = jsonroot['data']

            bounding_boxes = []
            one_hot_classes = []
            for box in boxes:
                xmin = float(box['x1'])
                ymin = float(box['y1'])
                xmax = float(box['x2'])
                ymax = float(box['y2'])
                bounding_box = [xmin, ymin, xmax, ymax]
                bounding_boxes.append(bounding_box)
                class_name = 'charactor'
                one_hot_class = self._to_one_hot(class_name)
                one_hot_classes.append(one_hot_class)

            image_name = ".".join(filename.split(".")[:-1])
            if len(images) > 0:
                idx = bisect(images, image_name) - 1
                image_name += exts[idx]
            else:
                image_name += '.png'

            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data

            count += 1
            if count % divisor == 0:
                print('{0}% - {1}...'.format(count // divisor, count))

    def _to_one_hot(self, name):
        one_hot_vector = [0] * self.num_classes
        if name == 'charactor':
            one_hot_vector[0] = 1
        else:
            print('unknown label: %s' % name)

        # if name == 'alphabet':
        #     one_hot_vector[0] = 1
        # elif name == 'number':
        #     one_hot_vector[1] = 1
        # elif name == 'symbol':
        #     one_hot_vector[2] = 1
        # else:
        #     print('unknown label: %s' %name)

        return one_hot_vector

## example on how to use it
# import pickle
# data = XML_preprocessor('VOC2007/Annotations/').data
# pickle.dump(data,open('VOC2007.p','wb'))

