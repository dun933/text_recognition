import json, cairocffi as cairo, random
import itertools
import multiprocessing as mp, logging
import math, cv2
from fontTools.ttLib import TTFont
import sys, os

from config.config_Vietnamese import Config_Vietnamese
from ImageGenTask import ImageGenTask_v3 as ImageGenTask
from config.ConfigManager import ConfigManager
from dataprovision_utils.Charset import Charset
from dataprovision_utils.image_augmentation_fns import apply_salt_and_pepper, apply_gaussian_blur, apply_motion_blur
from BackgroundItem import BackgroundType, BackgroundItem
from dataprovision_utils.bgfs_parsing import parse_bgfc_filepath
from dataprovision_utils.etc_fns import prepare_list_from_label_file, fetch_word_list_from_txt_file
from dataprovision_utils.gen_image_fns import generate_single_image_from_word_list_v1, get_statistics_for_font, is_visible_char, get_font_path, get_list_font_from_dir
from dataprovision_utils.CharBox import PredBox
from dataprovision_utils.voc_xml import save_xml

from tqdm import tqdm

random.seed()
dl = logging.getLogger("debug")
pool = None

class GenTaskManager:
    def __init__(self, bgimg_json,corpus_file, n_cpu=10):
        print('GenTaskManager.Init')
        self.bgimg_json=bgimg_json
        self.corpus_file=corpus_file
        self.configmgr = Config_Vietnamese
        self.gen_lowres = self.configmgr.gen_lowres
        self.init_task_list()
        self._init_word_list()
        self._init_char_list_dict()
        self._init_char_stats_for_height()
        self.provision_index = 0
        self._init_cpu_num(n_cpu)

    def init_task_list(self):
        print('GenTaskManager.init_task_list')
        bg_font_settings = parse_bgfc_filepath(self.bgimg_json)
        # tuan anh config font slant
        font_slant_options = [cairo.FONT_SLANT_NORMAL, cairo.FONT_SLANT_ITALIC, cairo.FONT_SLANT_OBLIQUE]
        font_weight_options = [cairo.FONT_WEIGHT_NORMAL, cairo.FONT_WEIGHT_BOLD]
        self.english_font_typeface_options = get_list_font_from_dir(self.configmgr.font_dir)
        font_rotate_options = [0]

        # create task settings
        params = {}
        params['bg_font_settings'] = bg_font_settings
        params['font_slant_options'] = font_slant_options
        params['font_weight_options'] = font_weight_options
        params['font_rotate_options'] = font_rotate_options
        params['english_font_typeface_options'] = self.english_font_typeface_options
        params['font_AA_options'] = [True, False]

        self.task_setting_list = self._init_task_settings_from_args(**params)

        # shuffle the task_setting_list
        random.shuffle(self.task_setting_list)
        dl.debug("task_setting_list size={}".format(len(self.task_setting_list)))

    def _init_task_settings_from_args(self, **params):
        print('GenTaskManager._init_task_settings_from_args')
        bg_font_settings = params['bg_font_settings']
        font_slant_options = params['font_slant_options']
        font_weight_options = params['font_weight_options']
        font_rotate_options = params['font_rotate_options']
        font_AA_enable_options = params['font_AA_options']

        english_font_typeface_options = params['english_font_typeface_options']
        gen_task_list = []
        self.font_typeface_options = set()

        for bg_font_setting in bg_font_settings:
            bgitem = bg_font_setting['bg_item']
            for font_slant, font_typeface, font_weight, font_rotate, font_AA_enable in \
                    itertools.product(font_slant_options, english_font_typeface_options, font_weight_options,
                                      font_rotate_options, font_AA_enable_options):
                genparams = {}
                genparams['bgitem'] = bgitem
                genparams['font_slant'] = font_slant
                genparams['font_weight'] = font_weight
                genparams['font_rotate'] = font_rotate
                genparams['font_AA_enable'] = font_AA_enable
                genparams['font_typeface'] = font_typeface

                task = ImageGenTask(**genparams)
                gen_task_list.append(task)
                self.font_typeface_options.add(font_typeface)

        return gen_task_list

    def _init_word_list(self):
        print('GenTaskManager._init_word_list')
        if self.corpus_file is None:
            self.word_list = []
        else:
            self.word_list = fetch_word_list_from_txt_file(self.corpus_file, mode=2)

    def _init_char_list_dict(self):
        print('GenTaskManager._init_char_list_dict')
        alphabet_char_filepath = self.configmgr.alphabet_char_filepath
        vietnamese_char_filepath = self.configmgr.vietnamese_char_filepath
        symbol_char_filepath = self.configmgr.symbol_char_filepath
        number_char_filepath = self.configmgr.number_char_filepath

        alphabet_char_list = prepare_list_from_label_file(alphabet_char_filepath)
        vn_char_list = prepare_list_from_label_file(vietnamese_char_filepath)
        symbol_char_list = prepare_list_from_label_file(symbol_char_filepath)
        number_char_list = prepare_list_from_label_file(number_char_filepath)

        self.char_list_dict = {
            "alphabet": alphabet_char_list,
            "vietnam": vn_char_list,
            "symbol": symbol_char_list,
            "number": number_char_list
        }

        dl.debug("char_list_dict: {}".format(self.char_list_dict))

    def _init_char_stats_for_height(self):
        print('GenTaskManager._init_char_stats_for_height')
        chars = []
        for _, v in self.char_list_dict.items():
            chars.extend(v)
        self.char_stats_dict = dict()

        for f in self.english_font_typeface_options:
            self.char_stats_dict[f] = dict()
            for size in [self.configmgr.font_min_size, self.configmgr.font_max_size]:
                mean_x_bearing, std_x_bearing, mean_y_bearing, std_y_bearing, \
                mean_width, std_width, mean_height, std_height = get_statistics_for_font(chars, f, size)
                self.char_stats_dict[f][size] = {"mean_x_bearing": mean_x_bearing,
                                                 "std_x_bearing": std_x_bearing,
                                                 "mean_y_bearing": mean_y_bearing,
                                                 "std_y_bearing": std_y_bearing,
                                                 "mean_width": mean_width,
                                                 "std_width": std_width,
                                                 "mean_height": mean_height,
                                                 "std_height": std_height}

    def _init_cpu_num(self, n_cpu=10):
        print('GenTaskManager._init_cpu_num')
        cpucount = os.cpu_count()
        cpu_use = cpucount - 4  # leave out some cpus for other processes to use
        cpu_use = min(cpu_use, n_cpu)
        if cpu_use <= 0:
            raise Exception("not enough cpus")

        self.pool_cpu_num = cpu_use
        print("# of CPUs: {}".format(self.pool_cpu_num))
        dl.debug("# of CPUs: {}".format(self.pool_cpu_num))

    def getdata(self, batch_size=10, repeat=1, save_online=False, output_dir=".", char_gen=False, language=0):
        """
        launches threads that will generate random data and returns the generated data
        :param batch_size: requested batch size
        :type batch_size: int
        :param repeat: # of images per batch
        :type repeat: int
        :returns: batch data that has been successfully generated
        """
        print('GenTaskManager.getdata')
        epoch_end_signal = False
        start_index = self.provision_index
        end_index = start_index + batch_size

        if end_index >= len(self.task_setting_list):
            end_index = len(self.task_setting_list)
            epoch_end_signal = True
            self.provision_index = 0
        else:
            self.provision_index = end_index

        fetched_task_list = self.task_setting_list[start_index: end_index]

        unsupported_char = {}
        charset = Charset(vietnam_file_path=self.configmgr.vietnamese_char_filepath,
                          symbol_file_path=self.configmgr.symbol_char_filepath, bg_symbol_file_path=self.configmgr.background_char_filepath)
        charlist = charset.get_all_char_list()
        for font_name in self.font_typeface_options:
            unsupported_char[font_name] = set()
            font_path = get_font_path(font_name, self.configmgr.font_dir)
            if font_path is None or len(font_path) < 4:
                print("Font path error: {}".format(font_name))
                continue

            if font_path[-3:] not in ['ttf', 'TTF']:
                font = TTFont(font_path, fontNumber=0)
            else:
                font = TTFont(font_path)
            for char in charlist:
                if is_visible_char(chr(char), font=font) is False:
                    unsupported_char[font_name].add(char)
        #print('GenTaskManager.getdata.unsupported char:', unsupported_char)
        random.shuffle(fetched_task_list)
        pool = mp.Pool(self.pool_cpu_num)
        mp_manager = mp.Manager()
        shared_queue = mp_manager.Queue()

        generate_single_image_fn = thread_gen_bunch_of_images

        bunch_size = 50
        print("# of images: {}".format(len(fetched_task_list) * repeat))
        print("# of images per bunch: {}".format(bunch_size * repeat))
        niter = math.ceil(len(fetched_task_list) / bunch_size)
        print("Loading {} of process pool.".format(niter))

        if self.word_list is None or len(self.word_list) == 0:
            print("Charlist generator")
        if (char_gen):
            results = [pool.apply_async(generate_single_image_fn,
                                        (list(range(it * bunch_size, (it + 1) * bunch_size)), fetched_task_list[it * bunch_size:(it + 1) * bunch_size],
                                         shared_queue, save_online, repeat, output_dir,
                                         self.char_list_dict, self.configmgr)) for it in tqdm(range(niter))]
        else:
            kw = {"unsupported_char": unsupported_char}
            results = [pool.apply_async(generate_single_image_fn,
                                        (list(range(it * bunch_size, (it + 1) * bunch_size)), fetched_task_list[it * bunch_size:(it + 1) * bunch_size],
                                         shared_queue,
                                         self.word_list, charset, self.configmgr,
                                         self.char_stats_dict, save_online, repeat,
                                         output_dir,), kw) for it in tqdm(range(niter))]
        try:
            print("Generating images.")
            for r in tqdm(results):
                r.get()
        except TimeoutError:
            print("time out error")

        pool.close()

        # extract generated image and gt box list from shared_queue.
        image_list = []
        charboxlist_list = []

        while not shared_queue.empty():
            image, charboxlist = shared_queue.get()
            image_list.append(image)
            charboxlist_list.append(charboxlist)

        dl.debug("image_list size={}".format(len(image_list)))
        dl.debug("charboxlist_list size={}".format(len(charboxlist_list)))

        return image_list, charboxlist_list, epoch_end_signal

    def get_data_size(self):
        return len(self.task_setting_list)


def thread_gen_bunch_of_images(indices, tasks, shared_queue, word_list, charset, configmgr, char_stats_dict, save_online, repeat,
                               output_dir, **kwargs):
    print('GenTaskManager.thread_gen_bunch_of_images')
    for i in range(min(len(indices), len(tasks))):
        index = indices[i]
        task = tasks[i]
        for r in range(repeat):
            thread_gen_image_v3(index, task, shared_queue, word_list, charset, configmgr, char_stats_dict, save_online, r,
                                output_dir, **kwargs)


def thread_gen_image_v3(index, task, shared_queue, word_list, charset, configmgr, char_stats_dict, save_online, r, output_dir, **kwargs):
    #print('GenTaskManager.thread_gen_image_v3')

    font_weight = task.font_weight
    font_rotate = task.font_rotate
    font_slant = task.font_slant
    font_AA_enable = task.font_AA_enable
    font_typeface = task.font_typeface
    bgitem = task.bgitem
    bg_id = bgitem.id

    unsupported_char = kwargs["unsupported_char"]

    kw = {
        "img_width": configmgr.img_width,
        "img_height": configmgr.img_height,
        "font_min_size": configmgr.font_min_size,
        "font_max_size": configmgr.font_max_size,
        "word_list": word_list,
        "font_weight": font_weight,
        "font_slant": font_slant,
        "font_typeface": font_typeface,
        "font_AA_enable": font_AA_enable,
        "bgitem": bgitem,
        "remove_unknown": True,
        "remove_invisible": True,
        "char_stats_dict": char_stats_dict,
        "unsupported_char": unsupported_char,
        "charset": charset
    }

    image, charbox_list, font_size = generate_single_image_from_word_list_v1(**kw)

    result = (image, charbox_list)
    if save_online:
        #print('GenTaskManager.thread_gen_image_v3.save_online')
        basename = "{}".format(repr(task))
        basename = basename.replace('None',str(font_size))
        #print(basename+'\n')
        img_filename = "{}.png".format(basename)
        img_dir = os.path.join(output_dir, "images")
        annot_dir = os.path.join(output_dir, "annots")
        xml_dir = os.path.join(output_dir, "XML")
        img_filepath = os.path.join(img_dir, img_filename)

        charbox_json_list = []
        for charbox in charbox_list:
            charbox_json_list.append(charbox.export_to_json())

        cv2.imwrite(img_filepath, image)

        # Save JSON file
        savejson = {"data": charbox_json_list}
        json_filename = "{}.json".format(basename)
        json_filepath = os.path.join(annot_dir, json_filename)

        # with open(json_filepath, 'w') as fd:
        #     json.dump(savejson, fd)
        with open(json_filepath, 'w', encoding='utf-8') as fd:
            json.dump(savejson, fd, ensure_ascii=False)
    else:
        shared_queue.put(result)

def save_to_txt(filepath, charboxs, img_width, img_height):
    f = open(filepath, "x")
    for charbox in charboxs:
        f.write("{} {} {} {} {}\n".format(charbox.char_value, int(round(charbox.x1 * img_width)),
                                          int(round(charbox.y1 * img_height)),
                                          int(round((charbox.x2 - charbox.x1) * img_width)),
                                          int(round((charbox.y2 - charbox.y1) * img_height))))
    f.close()

