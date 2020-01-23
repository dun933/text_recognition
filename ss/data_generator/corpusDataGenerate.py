from ImageGenSaver import ImageGenSaver
from time import time
from datetime import datetime
from config.config_Vietnamese import Config_Vietnamese
from bgfs_creation.create_json_from_cropped_bgimg import create_json
from corpus.corpus_utils import gen_final_corpus
import os, sys

class writer:
    def __init__(self, *writers):
        self.writers = writers
    def write(self, text):
        for w in self.writers:
            w.write(text)
    def flush(self):
        pass

if __name__ == '__main__':
    gen_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
    output_dir = Config_Vietnamese.output_dir+ '/corpus_' + str(Config_Vietnamese.num_data_generate) + '_' + gen_time
    os.makedirs(output_dir)
    # saved = sys.stdout
    # log_file = os.path.join(output_dir, "train.log")
    # f = open(log_file, 'w')
    # sys.stdout = writer(sys.stdout, f)

    #Gen json file
    bgimg_json = Config_Vietnamese.bgimg_json
    if(bgimg_json==''):
        bgimg_json = create_json(Config_Vietnamese.bgimg_dir)

    #Gen corpus file
    corpus_file = Config_Vietnamese.corpus_file
    if(corpus_file==''):
        corpus_file = gen_final_corpus('corpus')

    igs = ImageGenSaver(bgimg_json, corpus_file, output_dir, Config_Vietnamese.num_thread, remake=True)

    # sys.stdout = saved
    # f.close()
    print("Begin generate image")
    stime = time()
    igs.gen_and_save(Config_Vietnamese.num_data_generate, 1, save_online=True)
    etime = time()
    print("elapsed time: {}".format(etime - stime))
    print("done")

