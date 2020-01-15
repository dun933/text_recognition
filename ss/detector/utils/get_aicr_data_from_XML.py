import numpy as np
import os
from bisect import bisect
from xml.etree import ElementTree

class XML_preprocessor(object):

    def __init__(self, data_path, image_path = None):
        self.path_prefix = data_path
        self.num_classes = 3
        self.data = dict()
        self.path_image = image_path
        self._preprocess_XML()
        
    def _get_key(self, item) :
        return item[0]
    
    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        images = []
        exts = []
        if self.path_image is not None:
            tmp = os.listdir(self.path_image)            
            tmp = [os.path.splitext(filenm) for filenm in tmp]
            tmp = sorted(tmp, key=self._get_key)
            images, exts = zip(*tmp)
            
        for filename in filenames:
            #print(self.path_prefix + filename)
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                if  object_tree.find('name') is not None:
                    for bounding_box in object_tree.iter('bndbox'):
                        xmin = float(bounding_box.find('xmin').text)/width
                        ymin = float(bounding_box.find('ymin').text)/height
                        xmax = float(bounding_box.find('xmax').text)/width
                        ymax = float(bounding_box.find('ymax').text)/height
                    bounding_box = [xmin,ymin,xmax,ymax]
                    bounding_boxes.append(bounding_box)
                    class_name = object_tree.find('name').text
                    one_hot_class = self._to_one_hot(class_name)
                    one_hot_classes.append(one_hot_class)
            
            image_name = root.find('filename').text
            if len(images) > 0 :
                idx = bisect(images, image_name) - 1
                image_name += exts[idx]
            else :
                image_name += '.png'

            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data

    def _to_one_hot(self,name):
        one_hot_vector = [0] * self.num_classes
        if name == 'hangul':
            one_hot_vector[0] = 1
        elif name == 'alphabet':
            one_hot_vector[1] = 1
        elif name == 'num_symbol':
            one_hot_vector[2] = 1
        else:
            print('unknown label: %s' %name)

        return one_hot_vector

## example on how to use it
# import pickle
# data = XML_preprocessor('VOC2007/Annotations/').data
# pickle.dump(data,open('VOC2007.p','wb'))

