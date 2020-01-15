import configparser, json, os


class ConfigManager(object):

    def __init__(self, config_filename):
        config = configparser.ConfigParser()
        config.read(config_filename, encoding="utf-8")

        self._config_reader = config

        # classifier 1
        self.input_size_classifier1 = tuple(json.loads(config.get('common_classifier1', 'input_size')))
        self.batch_size_classifier1 = config.getint('common_classifier1', "batch_size")
        self.gpu_num_classifier1_train = config.getint('train_classifier1', "gpu_num")
        self.train_classifier1_dir=config.get('train_classifier1',"train_dir")
        self.val_classifier1_dir=config.get('train_classifier1',"val_dir")
        self.save_classifier1_dir=config.get('train_classifier1',"save_dir")

        self.gpu_num_classifier1_infer = config.getint('infer_classifier1', "gpu_num")
        self.test_classifier1_dir=config.get('infer_classifier1',"test_dir")
        self.weight_path_classifier1 = config.get('infer_classifier1', 'weight_path')

        # classifier 2
        self.input_size_classifier2 = tuple(json.loads(config.get('common_classifier2', 'input_size')))
        self.batch_size_classifier2 = config.getint('common_classifier2', "batch_size")
        self.group_classifier2=config.get('common_classifier2',"group")
        self.class_list_A=config.get('common_classifier2','class_list_A')
        self.class_list_alp=config.get('common_classifier2','class_list_alp')
        self.class_list_E=config.get('common_classifier2','class_list_E')
        self.class_list_I=config.get('common_classifier2','class_list_I')
        self.class_list_num=config.get('common_classifier2','class_list_num')
        self.class_list_O=config.get('common_classifier2','class_list_O')
        self.class_list_sym=config.get('common_classifier2','class_list_sym')
        self.class_list_U=config.get('common_classifier2','class_list_U')
        self.class_list_Y=config.get('common_classifier2','class_list_Y')
        self.ignore_class_list=config.get('common_classifier2',"ignore_class_list")

        self.gpu_num_classifier2_train = config.getint('train_classifier2', "gpu_num")
        self.train_classifier2_dir=config.get('train_classifier2',"train_dir")
        self.val_classifier2_dir=config.get('train_classifier2',"val_dir")
        self.save_classifier2_dir=config.get('train_classifier2',"save_dir")

        self.gpu_num_classifier2_infer = config.getint('infer_classifier2', "gpu_num")
        self.test_classifier2_dir=config.get('infer_classifier2',"test_dir")
        self.weight_path_classifier2 = config.get('infer_classifier2', 'weight_path')

        # classifier 12
        self.input_size_classifier12 = tuple(json.loads(config.get('common_classifier12', 'input_size')))
        self.batch_size_classifier12 = config.getint('common_classifier12', "batch_size")
        self.gpu_num_classifier12_train = config.get('train_classifier12', "gpu_num")
        self.train_classifier12_dir = config.get('train_classifier12', "train_dir")
        self.val_classifier12_dir = config.get('train_classifier12', "val_dir")
        self.save_classifier12_dir = config.get('train_classifier12', "save_dir")
        self.pre_trained_weight = config.get('train_classifier12', "pre_trained")

        self.gpu_num_classifier12_infer = config.getint('infer_classifier12', "gpu_num")
        self.test_classifier12_dir = config.get('infer_classifier12', "test_dir")
        self.weight_path_classifier12 = config.get('infer_classifier12', 'weight_path')
