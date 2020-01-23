
import os, cv2

def get_list_file_in_folder(dir, ext='jpg'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def convert_jpg2png(src_dir,dst_dir):
    list_file=get_list_file_in_folder(src_dir)
    for file in list_file:
        print(file)
        src_path=os.path.join(src_dir,file)
        dst_path=os.path.join(dst_dir,file.replace('.jpg','.png'))
        img =cv2.imread(src_path)
        cv2.imwrite(dst_path,img)



if __name__ == "__main__":
    src='/home/duycuong/PycharmProjects/research_py3/text_recognition/ss/data_generator/data/bg_img_jpg'
    dst='/home/duycuong/PycharmProjects/research_py3/text_recognition/ss/data_generator/data/bg_img_png'
    convert_jpg2png(src,dst)