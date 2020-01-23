import os, json
class ConfigManager:
    def __init__(self, configfilepath):
        print('ConfigManager.Init')
        if not os.path.exists(configfilepath):
            raise FileNotFoundError("{} not found".format(configfilepath))

        with open(configfilepath, 'r') as fd:
            configjson = json.load(fd)

        self.alphabet_char_filepath = configjson["alphabet_char_filepath"]
        self.vietnamese_char_filepath = configjson["vietnamese_char_filepath"]
        self.symbol_char_filepath = configjson["symbol_char_filepath"]
        self.number_char_filepath = configjson["number_char_filepath"]
        self.background_char_filepath = configjson["background_char_filepath"]

        self.bgimg_filepath = configjson["bgimg_filepath"]
        self.corpus_filepath = configjson.get("corpus_filepath", None)

        self.font_min_size = configjson["font_min_size"]
        self.font_max_size = configjson["font_max_size"]
        self.img_width = configjson["img_width"]
        self.img_height = configjson["img_height"]
        self.do_augmentation_flag = configjson["do_augmentation_flag"]
        self.draw_annotated_image = configjson["draw_annotated_image"]

        self.english_font_typeface_options = configjson["english_font_typeface_options"]
        self.vietnam_font_typeface_options = configjson["vietnam_font_typeface_options"]

        self.gen_lowres = configjson.get("gen_lowres", False)