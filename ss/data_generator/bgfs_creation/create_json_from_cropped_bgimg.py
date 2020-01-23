import json, os, datetime, sys, argparse

def overwrite_print(msg):
    sys.stdout.write("\r{}".format(msg))
    sys.stdout.flush()

def parse_args():
    parser = argparse.ArgumentParser(description='argparse for create_bgfs_for_cropped_bgimg.py')
    parser.add_argument('--imgdir', type=str, nargs=1, help="dir path containing the cropped bgs")
    args = parser.parse_args()
    return args

def create_json(cropped_img_dir):
    print('create json background file for folder:',cropped_img_dir)
    if not os.path.exists(cropped_img_dir):
        raise FileNotFoundError("{} not exist".format(cropped_img_dir))

    file_name = os.path.split(cropped_img_dir)
    output_json_filepath = "bgimg_{}.json".format(file_name[-1])
    output_json_filepath = os.path.join(cropped_img_dir, '../', output_json_filepath)

    if os.path.exists(output_json_filepath):
        print('json file',output_json_filepath, 'exist!',)
        return output_json_filepath
    imgfiles = os.listdir(cropped_img_dir)
    outjson = {}
    bgfontsettings = []
    count = 0

    for index, f in enumerate(imgfiles):
        imgfilepath = cropped_img_dir+'/'+ f
        bgfs = {
            "font-settings": [],
            "bg": imgfilepath,
            "bg_type": "image",
            "bg_id": str(count)
        }
        count += 1
        bgfontsettings.append(bgfs)
        overwrite_print("{}/{} done".format(index, len(imgfiles)))
    outjson["bg_font_settings"] = bgfontsettings

    with open(output_json_filepath, 'w') as fd:
        json.dump(outjson, fd)
    return output_json_filepath

if __name__ == "__main__":
    # cropped_img_dir = "/home/ubuntu/chadrick/detector_datagen/pure_python/cropped_bgs_190122_merge"
    args=parse_args()
    if args.imgdir is None:
        print("--imgdir argument missing")
    create_json(args.imgdir[0])
    print("done")

