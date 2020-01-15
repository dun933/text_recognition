from GenTaskManager import GenTaskManager
import os,json, shutil, cv2
from tqdm import tqdm

class ImageGenSaver:
    """
    Saves the generated data(image+annotation) to a directory.
    the generated data count will be tracked and used as filename when saving each data pair.
    """
    def __init__(self, bgimg_json, corpus_file, output_dir, n_cpu = 40, remake=True):
        print('ImageGenSaver.Init')
        self.gentaskmgr = GenTaskManager(bgimg_json, corpus_file, n_cpu)
        self.gen_count=0

        if remake:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)

        self.output_dir = output_dir
        
        self.img_dir = os.path.join(output_dir, "images")
        self.annot_dir = os.path.join(output_dir, "annots")
        self.xml_dir = os.path.join(output_dir, "XML")

        if remake:
            os.makedirs(self.img_dir)
            os.makedirs(self.annot_dir)
            os.makedirs(self.xml_dir)
    
    def gen_and_save(self, batch_size, repeat = 1, save_online = False):
        print('ImageGenSaver.gen_and_save')
        assert batch_size > 0
        img_list, charboxlist_list, epoch_end_signal= self.gentaskmgr.getdata(batch_size=batch_size, repeat=repeat, save_online=save_online, output_dir=self.output_dir)

        received_count = len(img_list)
        # save the received data
        print("Saving images.")
        if not save_online:
            print('ImageGenSaver.gen_and_save.Not save_online')
            for index, (img, charboxlist) in tqdm(enumerate(zip(img_list, charboxlist_list))):
                basename = "{:07}".format(self.gen_count)
                img_filename = "{}.png".format(basename)
                img_filepath = os.path.join(self.img_dir, img_filename )
                cv2.imwrite(img_filepath, img)

                # convert charlist_list to json object
                charbox_json_list=[]

                for charbox in charboxlist:
                    charbox_json_list.append(charbox.export_to_json())

                savejson={"data": charbox_json_list}

                json_filename = "{}.json".format(basename)
                json_filepath = os.path.join(self.annot_dir, json_filename)

                with open(json_filepath, 'w', encoding='utf-8') as fd:
                    json.dump(savejson, fd, ensure_ascii=False)

                self.gen_count +=1
