import numpy as np
import os
from bisect import bisect
# from xml.etree import ElementTree
import xml.etree.cElementTree as ElementTree
import json


# from lxml.etree import Element as ElementTree

class Json_preprocessor(object):

    def __init__(self, data_path, image_path=None, num_classes=4, remove_unused=False):
        self.path_prefix = data_path
        self.num_classes = num_classes
        self.data = dict()
        self.path_image = image_path
        self._get_file_name(remove_unused)

        self._preprocess_json()


    def _get_file_name(self, remove_unused):
        if remove_unused:
            img_path = os.path.join(self.path_prefix, "../images")
            print(img_path)
            filenames = os.listdir(self.path_prefix)
            img_names = os.listdir(img_path)
            filenames.sort()
            img_names.sort()
            self.file_names = []

            iter_img = 0
            for iter_file in range(len(filenames)):
                file = filenames[iter_file]
                img = file[0:-4]
                img += 'png'
                while True:
                    if img == img_names[iter_img]:
                        self.file_names.append(file)
                        iter_img += 1
                        break
                    elif img > img_names[iter_img]:
                        iter_img += 1
                    elif img < img_names[iter_img]:
                        break

        else:
            self.file_names = os.listdir(self.path_prefix)

        # exit()
    def _get_key(self, item):
        return item[0]

    def _preprocess_json(self):
        # self.file_names = os.listdir(self.path_prefix)
        images = []
        exts = []
        if self.path_image is not None:
            tmp = os.listdir(self.path_image)
            tmp = [os.path.splitext(filenm) for filenm in tmp]
            tmp = sorted(tmp, key=self._get_key)
            images, exts = zip(*tmp)



        print('total : ' + str(len(self.file_names)))
        count = 0
        divisor = len(self.file_names) // 100
        for filename in self.file_names:
            with open(os.path.join(self.path_prefix, filename), 'r') as fp:
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
                class_name = box['class_type']
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
        # if name == 'hangul':
        #     one_hot_vector[0] = 1
        # elif name == 'alphabet':
        #     one_hot_vector[1] = 1
        # elif name == 'number':
        #     one_hot_vector[2] = 1
        # elif name == 'symbol':
        #     one_hot_vector[3] = 1
        # else:
        #     print('unknown label: %s' %name)

        if name == 'vietnam':
            one_hot_vector[0] = 1
        elif name == 'alphabet':
            one_hot_vector[0] = 1 #2
        elif name == 'number':
            one_hot_vector[0] = 1
        elif name == 'symbol':
            one_hot_vector[0] = 1
        else:
            one_hot_vector[0] = 1
            # print('unknown label: %s' % name)
        return one_hot_vector

## example on how to use it
# import pickle
# data = XML_preprocessor('VOC2007/Annotations/').data
# pickle.dump(data,open('VOC2007.p','wb'))

