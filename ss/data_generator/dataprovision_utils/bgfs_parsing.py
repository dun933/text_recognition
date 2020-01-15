import json, os
from .ColorParser import ColorParser

try:
    from BackgroundItem import BackgroundType, BackgroundItem
except ImportError:
    from ..BackgroundItem import BackgroundType, BackgroundItem

colorparser = ColorParser()

def parse_bg_font_settings_root(root):
    print('bgfs_parsing.parse_bg_font_settings_root')
    bg_font_settings = root['bg_font_settings']
    output_list=[]

    for setting in bg_font_settings:
        bg = setting['bg']
        bg_type = setting['bg_type']
        if bg_type=="image":
            bg_id = setting['bg_id']
            enum_bg_type = BackgroundType.IMAGE
            bgitem = BackgroundItem(enum_bg_type, bg, id=bg_id)
        elif bg_type=="solidcolor":
            enum_bg_type = BackgroundType.SOLIDCOLOR
            bgitem = BackgroundItem(enum_bg_type, bg)
            bg_solidcolor_type = colorparser.is_color_string(bg)
            if bg_solidcolor_type == "keyword":
                cairo_format = colorparser.convert_keyword_to_cairo_format(bg)
                bgitem.set_cairo_format(cairo_format)
                idval = colorparser.convert_cairo_format_html_code_str(cairo_format)
                bgitem.set_id(idval)
            elif bg_solidcolor_type == "html_code":
                cairo_format = colorparser.convert_html_format_to_cairo_format(bg)
                bgitem.set_cairo_format(cairo_format)
                bgitem.set_id(bg)
            else:
                raise Exception("unknown value")
        else:
            raise Exception("{} is invalid value".format(bg_type))
        # replace 'bg' key with bgitem
        setting['bg_item'] = bgitem
        font_settings = setting['font-settings']
        for fs in font_settings:
            font_basecolor = fs['font-basecolor']
            fs['font-basecolor'] = colorparser.convert(font_basecolor)
            font_outlinecolors = fs['font-outlinecolors']
            cvt_oc_list=[]
            for oc in font_outlinecolors:
                if oc is not None:
                    cvt_oc_list.append(colorparser.convert(oc))
                else:
                    cvt_oc_list.append(oc)

            fs['font-outlinecolors']= cvt_oc_list
        output_list.append(setting)

    return output_list

def parse_bgfc_filepath(bgfc_filepath):
    print('bgfs_parsing.parse_bgfc_filepath:'+bgfc_filepath)
    if bgfc_filepath is None:
        return []
    if not os.path.exists(bgfc_filepath):
        raise FileNotFoundError("{} not found".format(bgfc_filepath))

    with open(bgfc_filepath, 'r') as fd:
        bgfcjson = json.load(fd)

    bgfc_list = parse_bg_font_settings_root(bgfcjson)

    return bgfc_list