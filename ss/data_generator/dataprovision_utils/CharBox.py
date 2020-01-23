from enum import Enum
class TextFailure(Enum):
    EXCEED_TOP_LIMIT = 1
    EXCEED_BOTTOM_LIMIT = 2
    EXCEED_LEFT_LIMIT= 3
    EXCEED_RIGHT_LIMIT = 4
    OVERLAP_WITH_EXISTING_BOX = 5

class Box:
    def __init__(self, x1,y1,x2,y2):
        self.x1 = x1 
        self.x2 = x2 
        self.y1 = y1 
        self.y2 = y2

class PredBox:
    
    def __init__(self, classid, conf, x1,y1,x2,y2):
        """
        x1,y1,x2,y2 are img_w, img_h normalized
        """
        self.classid = classid
        self.conf = conf
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2 

    def area(self):
        return (self.y2 - self.y1) * (self.x2 - self.x1)

class CharBox(Box):
    def __init__(self, x1, y1, x2, y2, char_value, class_type):

        self.class_type = class_type
        self.char_value = char_value
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
    
    def export_to_json(self):
        retjson={
            "class_type": self.class_type,
            "char" : self.char_value,
            "x1" : self.x1,
            "x2" : self.x2,
            "y1" : self.y1,
            "y2": self.y2
        }

        return retjson
    
    @staticmethod
    def create_from_json(jsonobj):
        x1 = jsonobj["x1"]
        y1 = jsonobj["y1"]
        x2 = jsonobj["x2"]
        y2 = jsonobj["y2"]

        coords = [x1,y1,x2,y2]

        for x in coords:
            assert x <= 1.0 , "found coord that is > 1.0 : {}. full jsonobj: {}".format(x, jsonobj)

        class_type = jsonobj["class_type"]
        char_value = jsonobj["char"]

        return CharBox(x1,y1,x2,y2, char_value, class_type)

def is_overlapping(box1, box2):
    overlap_x1 = max(box1.x1, box2.x1)
    overlap_y1 = max(box1.y1 , box2.y1)

    overlap_x2 = min(box1.x2 , box2.x2)
    overlap_y2 = min(box1.y2, box2.y2)

    w = overlap_x2 - overlap_x1 
    h = overlap_y2 - overlap_y1

    if w<=0 or h<=0:
        return False
    else:
        return True