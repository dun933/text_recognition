
class Config_Vietnamese:
    #background image dir
    bgimg_json='data/bgimg_bg_img_png.json'
    #bgimg_json=''
    bgimg_dir='data/bg_img_png'

    #corpus list
    corpus_file='corpus/final_corpus_14Jan.txt'
    output_dir='outputs'

    font_dir='data/fonts_vn'

    num_data_generate = 300000
    num_thread = 10
    # parameters
    gauss_blur = 0.2
    salt_and_pepper = 0.2
    low_res_by_resizing = 0.1
    invert = 0.05
    jpeg_compression = 0.1

    #
    alphabet_char_filepath = "./char_list/alphabet.txt"
    vietnamese_char_filepath = "./char_list/vietnamese.txt"
    symbol_char_filepath = "./char_list/symbols.txt"
    number_char_filepath = "./char_list/numbers.txt"
    background_char_filepath = "./char_list/background.txt"
    font_min_size= 18
    font_max_size= 70
    img_width= 320
    img_height= 320
    do_augmentation_flag= True
    draw_annotated_image= False
    gen_lowres = False

    english_font_typeface_options= [
        "Times\\-Narrow",
        "Times_New_Roman",
        "DejaVuSerif",
        "Noto Sans CJK KR",
        "SansSerif",
        "DejaVu Sans",
        "Calibri",
        "Helvetica\\-Narrow",
        "Arial\\-Rounded",
        "GeoSlab703\\-Extra",
        "Commerce",
        "Georgia Ref",
        "Arrus\\-Black",
        "CenturionOld",
        "Isadora",
        "FreeSans",
        "FreeMono",
        "Tahoma",
        "Bauhaus\\-Medium",
        "BankGothic\\-Medium",
        "GillSans",
        "Souvenir",
        "NewsGothic",
        "Courier New",
        "Garamond",
        "OfficinaSerif",
        "Arial",
        "Commerce\\-Condensed",
        "Bookman",
        "Imago\\-ExtraBold",
        "Microsoft Sans Serif",
        "Candara\\-Viet",
        "FreeSerif",
        "NewCentury\\-Narrow",
        "Perpetua",
        "OfficinaSans",
        "Casablanca",
        "Roboto Condensed",
        "Noto Sans Mono CJK KR",
        "Verdana",
        "Palatino"
    ]
