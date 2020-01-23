
class ImageGenTask_v3:
    """
    derived from ImageGenTask_v3. remove fontbasecolor and outlinecolor
    """
    def __init__(self, **kwargs):
        self.bgitem = kwargs.get('bgitem', None)
        self.font_slant = kwargs.get('font_slant', None)
        self.font_weight = kwargs.get('font_weight', None)
        self.font_rotate = kwargs.get('font_rotate', None)
        self.font_size = kwargs.get('font_size', None)
        self.font_typeface = kwargs.get('font_typeface', None)
        self.font_AA_enable = kwargs.get('font_AA_enable', True)

    def export_to_json_obj(self):
        root = {}
        root['bgitem'] = self.bgitem.export_to_json_obj() #json object
        root['font_slant'] = self.font_slant # int
        root['font_weight'] = self.font_weight # int
        root['font_rotate'] = self.font_rotate # float or int
        root["font_typeface"] = self.font_typeface
        root["font_AA_enable"] = self.font_AA_enable

        return root

    def __repr__(self):
        name = ""
        for k, v in self.__dict__.items():
            if k == "bgitem":
                img_name = v.imgfilepath.split("/")[-1].replace('.png','')
                name = "{}_{}".format(k, img_name)
            else:
                name = "-".join([name,"{}_{}".format(k.replace('font_',''), v)])
        return name



