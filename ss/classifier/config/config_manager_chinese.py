import configparser, json, os

class ConfigManager(object):

    def __init__(self, config_filename):
        config = configparser.ConfigParser()
        config.read(config_filename, encoding="utf-8")

        self._config_reader = config

         # classifier 2
        self.input_size_classifier = tuple(json.loads(config.get('common_classifier', 'input_size')))
        self.batch_size_classifier = config.getint('common_classifier', "batch_size")

        self.gpu_num_classifier_train = config.get('train_classifier', "gpu_num")
        self.train_classifier_dir=config.get('train_classifier', "train_dir")
        self.val_classifier_dir=config.get('train_classifier', "val_dir")
        self.save_classifier_dir=config.get('train_classifier', "save_dir")
        self.restore_ckpt = config.getboolean('train_classifier', 'restore_ckpt')
        self.restore_ckpt_path = config.get('train_classifier', "restore_ckpt_path")

        self.gpu_num_classifier_infer = config.get('infer_classifier', "gpu_num")
        self.test_classifier_dir=config.get('infer_classifier',"test_dir")
        self.weight_path_classifier = config.get('infer_classifier', 'weight_path')
        self.model_path_classifier = config.get('infer_classifier', 'model_path')
        self.class_map = config.get('infer_classifier', 'class_map')