import configparser, json, os


class ConfigManager(object):

    def __init__(self, config_filename):
        """
        생성자 함수
        객체 초기화 시 설정파일에서 값을 읽어옴

        Parameters
        ----------
        config_filename: 설정값이 저장된 ini파일 경로

        """
        config = configparser.ConfigParser()
        config.read(config_filename, encoding="utf-8")

        self._config_reader = config

        self.classes = json.loads(config.get('common', 'classes'))  # Detection 분류 클래스
        self.gpu_num = config.get('common', 'gpu_num')

        self.restore_ckpt_flag = config.getboolean('train', "restore_ckpt")
        self.restore_ckpt_path = config.get('train', "restore_ckpt_path")
        # self.dataset_basedir = config.get('train', "dataset_basedir")
        self.dataset_pickle = config.get('train', "dataset_pickle")
        self.prior_pkl_file = config.get('train', "prior_pkl_file")
        self.sub_log_dir = config.get('train', "sub_log_dir")

        #cuongnd
        self.train_dir=config.get('train',"train_dir")
        self.val_dir=config.get('train',"val_dir")
        self.save_dir=config.get('train',"save_dir")
        #end

        self.img_shape = tuple(json.loads(config.get('infer', 'img_shape')))  # 분할 처리된 이미지 shape
        self.img_font_idle_size = config.getint('infer', 'img_font_idle_size')  # 인식이 잘되는 이상적인 폰트 크기(높이)
        self.img_font_idle_size2 = config.getint('infer', 'img_font_idle_size2')  # 인식이 잘되는 이상적인 폰트 크기(높이)
        self.split_overap_size = config.getint('infer', 'split_overap_size')  # 원본 이미지 분할 시 overap 크기
        #self.iou_threshold = config.getfloat('infer', 'iou_threshold')

        self.ssd_weight_path = config.get('infer', 'ssd_weight_path')
        self.nms_algorithm = config.get('infer', 'NMS_ALGORITHM')

        self.infer_gpu_num = config.get('infer', 'infer_gpu_num')

        # self.confidence_threshold = config.getfloat('test', 'confidence_threshold')


