
class ColorParser:
    def __init__(self):
        print('ColorParser.Init')
        self.color_keywords = {
            'grey': "#727272",
            "gray": "#727272",
            "red": "#ff0000",
            "yellow": "#ffff00",
            "blue": "#0000ff",
            "green": "#00ff00",
            "black": "#000000",
            "white": "#ffffff",
        }
        self.mode = "cairo"

    def convert(self, rawinput):
        if self.check_is_html_color_code_format(rawinput):
            return self.convert_html_format_to_cairo_format(rawinput)
        else:
            return self.convert_keyword_to_cairo_format(rawinput)

    def convert_keyword_to_cairo_format(self, keyword):
        # will convert the given keyword to (r,g,b) tuple with range 0.0 - 1.0 so that it can be directly
        # used for cairo

        rawval = self.color_keywords.get(keyword, None)
        if rawval is None:
            raise Exception("no matching color keyword found for {}".format(keyword))

        if self.check_is_html_color_code_format(rawval):
            return self.convert_html_format_to_cairo_format(rawval)

    def convert_html_format_to_cairo_format(self, html_format_str):
        """
        convert html format(e.g #ffffff) to cairo format
        """
        html_format_r = html_format_str[1:3]
        html_format_g = html_format_str[3:5]
        html_format_b = html_format_str[5:7]
        # print("html_format_r = {}".format(html_format_r))

        int_r = int(html_format_r, 16)
        int_g = int(html_format_g,16)
        int_b = int(html_format_b, 16)
        # print("int_r={}".format(int_r))

        cairo_format_r = int_r / 255.0
        cairo_format_g = int_g / 255.0
        cairo_format_b = int_b / 255.0
        # print("cairo_format_r={}".format(cairo_format_r))

        return (cairo_format_r, cairo_format_g, cairo_format_b)


    def check_is_html_color_code_format(self, given_str):
        if len(given_str) == 7 and given_str[0]=='#':
            return True
        else:
            return False

    def is_color_string(self, rawinput):
        
        if self.check_is_html_color_code_format(rawinput):
            return "html_code"
        elif rawinput in self.color_keywords:
            return "keyword"
        else:
            return None

    def convert_cairo_format_html_code_str(self, cairo_format_input):
        """
        convert cairo format to html code 
        """
        cairo_r, cairo_g, cairo_b = cairo_format_input
        int_r = int(cairo_r * 255)
        int_g = int(cairo_g * 255)
        int_b = int(cairo_b * 255)

        hex_r = "{:02x}".format(int_r)
        hex_g = "{:02x}".format(int_g)
        hex_b = "{:02x}".format(int_b)

        return "{}{}{}".format(hex_r, hex_g, hex_b)

def convert_cairo_format_to_rgb255base(cairo_format_rgb):
    """
    convert cairo format rgb(normed by 1.0) to 255 size normalized rgb value
    python list of rgb with each value in 255 base int value
    """
    rgb_list = list(cairo_format_rgb)
    rgb_list = list(map(lambda x: int(255*x), rgb_list))
    return rgb_list

def convert_rgb255base_to_cairo_format(rgb255base_color):
    """
    convert 255base rgb color tuple/list into cairo format(normalized by 1.0) tuple
    """
    if isinstance(rgb255base_color, tuple):
        rgb255base_color = list(rgb255base_color)
    output_rgb = list(map(lambda x: float(x)/255 , rgb255base_color))
    return tuple(output_rgb)
    

    

        