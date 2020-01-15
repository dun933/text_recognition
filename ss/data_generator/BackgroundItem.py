import  os
from enum import Enum

class BackgroundType(Enum):
    SOLIDCOLOR=0
    IMAGE=1

class BackgroundItem:
    def __init__(self,bgtype,value,id=None):
        if bgtype not in BackgroundType:
            raise Exception
        if bgtype == BackgroundType.SOLIDCOLOR:
            self.type=bgtype
            self.keyword = value
        elif bgtype==BackgroundType.IMAGE:
            self.type = bgtype
            self.imgfilepath = value
            self.id=id
            if not os.path.exists(self.imgfilepath):
                raise Exception("{} img file not found".format(self.imgfilepath))
        else:
            raise Exception("unknown value detected")
    def set_cairo_format(self,cairoformat):
        self.cairo_format = cairoformat

    def set_id(self, idval):
        self.id = idval
    
    def export_to_json_obj(self):
        root={}
        root['type'] = self.type.value
        if self.type== BackgroundType.SOLIDCOLOR:
            root['value'] = self.keyword
        elif self.type == BackgroundType.IMAGE:
            root['value'] = self.imgfilepath
        else:
            raise Exception("unknown type detected")
        return root
    
    
