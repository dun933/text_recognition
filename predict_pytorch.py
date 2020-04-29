import numpy as np
import torch, cv2, math
import os, time
from detector_DB.concern.config import Configurable, Config
from BoundingBox import bbox
import argparse

# classifier
from classifier_CRNN.models.utils import strLabelConverter
from torch.autograd import Variable
import classifier_CRNN.models.crnn as crnn
import classifier_CRNN.models.utils as utils
from torchvision import transforms
from pre_processing.augment_functions import cnd_aug_randomResizePadding, cnd_aug_resizePadding
from torchvision.transforms import RandomApply, ColorJitter, RandomAffine, ToTensor, Normalize
from classifier_CRNN.utils.loader import NumpyListLoader, alignCollate
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from structure.model import SegDetectorModel
from structure.representers.seg_detector_representer import SegDetectorRepresenter
from structure.visualizers.seg_detector_visualizer import SegDetectorVisualizer

os.environ["PYTHONIOENCODING"] = "utf-8"

gpu = '0'
#gpu= None
img_path = 'detector_DB_train/datasets/invoices_28April/test_images/II-9_d.png'
output_dir = 'outputs'
# detector
detector_model = 'model_epoch_200_minibatch_297000_8Mar'
ckpt_path = 'detector_DB_train/outputs/' + detector_model

detector_model = 'model_epoch_571_minibatch_12000'
ckpt_path = 'detector_DB_train/outputs/train_2020-04-28_22-54/model' + detector_model

detector_box_thres = 0.315
polygon = False
visualize = False
img_short_side = 736  # 736

# classifier
classifier_ckpt_path = 'AICR_pretrained_59_Test_43.16_cer_0.227.pth'
classifier_width = 1000

classifier_height = 64
alphabet_path = 'config/char_246'
if classifier_height == 64:
    classifier_ckpt_path = 'classifier_CRNN/ckpt/AICR_SDV_30Mar_300_loss_1.25_cer_0.0076.pth'
    alphabet_path = 'config/char_246'
classifier_batch_sz = 16
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
fill_color = (255, 255, 255)
draw_text = True
debug = False
if debug:
    classifier_batch_sz = 1


def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint', default=ckpt_path)
    parser.add_argument('--result_dir', type=str, default=output_dir, help='path to save results')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=img_short_side,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=detector_box_thres,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--resize', action='store_true', help='resize')
    parser.add_argument('--visualize', default=visualize, help='visualize maps in tensorboard')
    parser.add_argument('--polygon', help='output polygons if true', default=polygon)
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    # initialize
    begin_init = time.time()
    detector, classifier = init_models(args, gpu=gpu)
    test_img = cv2.imread(img_path)
    end_init = time.time()
    print('Init models time:', end_init - begin_init, 'seconds')

    boxes_list = detector.inference(img_path, visualize)
    end_detector = time.time()
    print('Detector time:', end_detector - end_init, 'seconds')

    boxes_data, boxes_info, max_wh_ratio = get_boxes_data(test_img, boxes_list)
    end_get_boxes_data = time.time()
    print('Get boxes time:', end_get_boxes_data - end_detector, 'seconds')

    values = classifier.inference(boxes_data, max_wh_ratio)
    end_classifier = time.time()
    print('Classifier time:', end_classifier - end_get_boxes_data, 'seconds')
    print('\nTotal predict time:', end_classifier - end_init, 'seconds')

    for idx, box in enumerate(boxes_info):
        box.asign_value(values[idx])
    visualize_results(test_img, boxes_info, draw_text)
    end_visualize = time.time()
    print('Visualize time:', end_visualize - end_classifier, 'seconds')
    print('Done')


def visualize_results(img, boxes_info, text=False, inch=40):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(inch, inch)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.imshow(img, cmap='Greys_r')

    for box in boxes_info:
        if text:
            plt.text(box.xmin - 2, box.ymin - 4, box.value, fontsize=max(int(box.height / 3), 12),
                     fontdict={"color": 'r'})

        ax.add_patch(patches.Rectangle((box.xmin, box.ymin), box.width, box.height,
                                       linewidth=2, edgecolor='green', facecolor='none'))

    # plt.show()
    save_img_path = os.path.join(output_dir, img_path.split('/')[-1].split('.')[0] + '_visualized.jpg')
    print('Save image to', save_img_path)
    fig.savefig(save_img_path, bbox_inches='tight')


def init_models(args, gpu='0'):
    if gpu != None:
        print('Use GPU', gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    else:
        print('Use CPU')
    detector = Detector_DB(gpu=gpu, cmd=args)
    classifier = Classifier_CRNN(ckpt_path=classifier_ckpt_path, batch_sz=classifier_batch_sz,
                                 imgW=classifier_width, imgH=classifier_height, gpu=gpu, alphabet_path=alphabet_path)
    return detector, classifier


class Detector_DB:
    def __init__(self, gpu='0', cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.gpu = gpu
        self.args = cmd
        self.init_torch_tensor()
        self.init_model(self.args['resume'])
        self.model.eval()

        self.segRepresent = SegDetectorRepresenter()
        self.segVisualizer = SegDetectorVisualizer()

    def init_torch_tensor(self):
        if torch.cuda.is_available() and self.gpu != None:
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

    def init_model(self, path):
        self.model = SegDetectorModel(self.device, distributed=False, local_rank=0)
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(path, map_location=self.device)
        self.model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape

    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i, :, :].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")

    def inference(self, image_path, visualize=False):
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)
        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = self.model.forward(batch, training=False)
            output = self.segRepresent.represent(batch, _pred=pred, is_output_polygon=self.args['polygon'])
            if not os.path.isdir(self.args['result_dir']):
                os.mkdir(self.args['result_dir'])
            self.format_output(batch, output)
            boxes, _ = output
            boxes = boxes[0]

            if visualize:
                vis_image = self.segVisualizer.demo_visualize(image_path, output)
                cv2.imwrite(os.path.join(self.args['result_dir'],
                                         image_path.split('/')[-1].split('.')[0] + '_ ' + detector_model + '_ ' + str
                                         (detector_box_thres) + '.jpg'), vis_image)
            return boxes


class Classifier_CRNN:
    def __init__(self, ckpt_path='', gpu='0', batch_sz=16, workers=4, num_channel=3, imgW=256, imgH=64,
                 alphabet_path='config/char_246'):
        self.imgW = imgW
        self.imgH = imgH
        self.batch_sz = batch_sz
        alphabet = open(alphabet_path, encoding='utf-8').read().rstrip()
        nclass = len(alphabet) + 1
        self.image = torch.FloatTensor(batch_sz, 3, imgH, imgH)
        self.text = torch.IntTensor(batch_sz * 5)
        self.length = torch.IntTensor(batch_sz)
        if classifier_height == 32:
            self.model = crnn.CRNN32(imgH, num_channel, nclass, 256)
        else:
            self.model = crnn.CRNN64(imgH, num_channel, nclass, 256)
        if gpu != None and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.image = self.image.cuda()
        print('Classifier. Resumed from %s' % ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.converter = strLabelConverter(alphabet, ignore_case=False)
        self.image = Variable(self.image)
        self.text = Variable(self.text)
        self.length = Variable(self.length)
        self.workers = workers
        self.model.eval()

    def inference(self, img_list, max_wh_ratio):
        new_W = int(self.imgH * max_wh_ratio)
        print('New W', new_W)
        transform_test = transforms.Compose([
            # cnd_aug_randomResizePadding(imgH, imgW, min_scale, max_scale, fill=fill_color, train=False),
            cnd_aug_resizePadding(new_W, self.imgH, fill=fill_color, train=False),
            ToTensor(),
            Normalize(mean, std)
        ])

        val_dataset = NumpyListLoader(img_list, transform=transform_test)
        num_files = len(val_dataset)
        print('Classifier. Begin classify', num_files, 'boxes')
        values = []
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_sz,
            num_workers=self.workers,
            shuffle=False
        )

        val_iter = iter(val_loader)
        max_iter = len(val_loader)
        # print('Number of samples', num_files)
        # begin = time.time()
        with torch.no_grad():
            for i in range(max_iter):
                data = val_iter.next()
                cpu_images, cpu_texts, _ = data
                batch_size = cpu_images.size(0)
                utils.loadData(self.image, cpu_images)
                preds = self.model(self.image)
                preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
                raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
                values.extend(sim_pred)
                if debug:
                    print('\n   ', raw_pred)
                    print(' =>', sim_pred)
                    cv_img = cpu_images[0].permute(1, 2, 0).numpy()
                    cv_img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    cv2.imshow('result', cv_img_bgr)
                    cv2.waitKey(0)
        # end = time.time()
        # processing_time = end - begin
        # print('Processing time:', processing_time)
        # print('Speed:', num_files / processing_time, 'fps')
        return values


def crop_from_img_rectangle(img, left, top, right, bottom):
    extend_y = 6
    extend_x = 0
    top = max(0, top - extend_y)
    bottom = min(img.shape[0], bottom + extend_y)
    left = max(0, left - extend_x)
    right = min(img.shape[1], right + extend_x)
    if left >= right or top >= bottom or left < 0 or right < 0 or left >= img.shape[1] or right >= img.shape[1]:
        return True, None
    return False, img[top:bottom, left:right], left, top, right, bottom


def get_boxes_data(img, boxes):
    boxes_data = []
    boxes_info = []
    max_wh_ratio = 1
    total=0
    for box_loc in boxes:
        box_loc = np.array(box_loc).astype(np.int32).reshape(-1, 2)
        left = min(box_loc[0][0], box_loc[3][0])
        top = min(box_loc[0][1], box_loc[1][1])
        right = max(box_loc[1][0], box_loc[2][0])
        bottom = max(box_loc[2][1], box_loc[3][1])
        if (right - left) < 20 or (bottom - top) < 10:
            continue
        NG, box_data, new_left, new_top, new_right, new_bottom = crop_from_img_rectangle(img, left, top, right, bottom)
        if NG:
            continue
        wh_ratio = (new_right - new_left) / (new_bottom - new_top)
        total+=wh_ratio
        if wh_ratio > max_wh_ratio:
            max_wh_ratio = wh_ratio
        box_info = bbox(new_left, new_top, new_right, new_bottom)
        boxes_info.append(box_info)
        boxes_data.append(box_data)
    print('Max wh ratio:', max_wh_ratio, 'total',total)
    return boxes_data, boxes_info, max_wh_ratio


if __name__ == '__main__':
    main()
