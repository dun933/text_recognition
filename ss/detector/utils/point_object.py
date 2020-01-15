import numpy as np


class PointObject(object):

    def __init__(self, class_nm, confidence, zoom_ratio=1.0):
        self.class_nm = class_nm
        self.confidence = confidence
        self.zoom_ratio = zoom_ratio

        self.absoulte_coord = []
        self.tight_absolute_coord = []
        self.np_img = None
        self.char_value = ''

    def set_absolute_coord(self, coord):
        assert isinstance(coord, list)
        assert len(coord) == 4

        self.absoulte_coord = coord

    def setAbsoluteCoord(self, relative_coord, split_file):

        end_y, end_x, start_y, start_x = split_file.split('_')[::-1][0:4]
        # extra_coord = (np.asarray(relative_coord) / self.zoom_ratio).tolist()
        absoulte_coord = [round((float(relative_coord[0]) + float(start_x)) / self.zoom_ratio),
                          round((float(relative_coord[1]) + float(start_y)) / self.zoom_ratio),
                          round((float(relative_coord[2]) + float(start_x)) / self.zoom_ratio),
                          round((float(relative_coord[3]) + float(start_y)) / self.zoom_ratio)]

        self.absoulte_coord = absoulte_coord

    def setAbsoluteCoordByObj(self, relative_coord, split_coord):

        absoulte_coord = [round((float(relative_coord[0]) + float(split_coord[0])) / float(split_coord[4])),
                          round((float(relative_coord[1]) + float(split_coord[1])) / float(split_coord[4])),
                          round((float(relative_coord[2]) + float(split_coord[0])) / float(split_coord[4])),
                          round((float(relative_coord[3]) + float(split_coord[1])) / float(split_coord[4]))]

        self.absoulte_coord = absoulte_coord

    def setCropImgCoord(self, img):

        if len(img.shape) == 3:
            img = img.dot([0.299, 0.587, 0.114])

        self.np_img = img[self.absoulte_coord[1]:self.absoulte_coord[3],
                      self.absoulte_coord[0]:self.absoulte_coord[2]]

    def getCropImgCoord(self):
        return self.np_img

    def getAbsolute_coord(self):
        return self.absoulte_coord

    def setCharValue(self, char):
        self.char_value = char

    def toList(self):

        return [self.absoulte_coord[0],
                self.absoulte_coord[1],
                self.absoulte_coord[2],
                self.absoulte_coord[3],
                self.char_value,
                self.confidence,
                self.class_nm
                ]

    def __lt__(self, other):
        if self.absoulte_coord[1] < other.absoulte_coord[1]:
            return True
        elif self.absoulte_coord[1] == other.absoulte_coord[1]:
            return self.absoulte_coord[0] < other.absoulte_coord[0]
        else:
            return False