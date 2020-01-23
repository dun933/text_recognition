''' '''

import glob
import cv2
import numpy as np, logging, time

from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K

try:
    from model.model_builder import SSD_AICR
    #from model.model_builder_tuananh import SSD_AICR
    #from model.ssd_hector import SSD_AICR_v03
    from utils.ssd_utils import BBoxUtility
    from utils.point_object import PointObject

except ImportError:
    from aicr_dssd_train.model.model_builder import SSD_AICR
    #from aicr_dssd_train.model.model_builder_tuananh import SSD_AICR
    #from aicr_dssd_train.model.ssd_hector import SSD_AICR_v03
    from aicr_dssd_train.utils.ssd_utils import BBoxUtility
    from aicr_dssd_train.utils.point_object import PointObject

np.set_printoptions(suppress=True)
dl = logging.getLogger("debug")

class SSD_Predict:
    ''' This class provides the Prediction function of the SSD algorithm.

    # Arguments
        classes : Object detection class
        split_img_path : Split image of document to run predication
        weight_path : model weight file path
        confidence : confidence score of bounding box
        input_shape : shape of input image (width, height, channel)

    '''

    def __init__(self, classes=['hangul', 'alphabet', 'num_symbol'],
                 split_img_path='./splito/',
                 weight_path='./checkpoints/weights.2016-02-16_04-0.13.hdf5',
                 confidence=0.35,
                 input_shape=(320, 320, 3),
                 zoom_ratio=1.0,
                 batch_size=32,
                 nms_thresh=0.45,
                 NMS_ALGORITHM="NMS",
                 margin_threshold=5,
                 margin_weight=0.8):
        if classes is None:
            classes = ['hangul', 'alphabet', 'num_symbol']

        self.classes = classes
        self.split_img_path = split_img_path
        self.weight_path = weight_path
        self.confidence = confidence
        self.input_shape = input_shape
        self.zoom_ratio = zoom_ratio
        # Total class number = classes + backgraound
        self.NUM_CLASSES = len(classes) + 1
        # model loading

        self.model = SSD_AICR(self.input_shape, num_classes=self.NUM_CLASSES)
        self.model.load_weights(weight_path)
        self.bbox_util = BBoxUtility(self.NUM_CLASSES, nms_thresh=nms_thresh)

        self.batch_size = batch_size

        self.NMS_ALGORITHM = NMS_ALGORITHM

        self._init_marginal_box(margin_threshold, margin_weight)

    def _init_marginal_box(self, margin_threshold=5, margin_weight=0.8):
        self.margin_threshold = margin_threshold
        self.margin_weight = margin_weight

    def _realign_weight_of_marginal_box(self, conf, xmin, ymin, xmax, ymax):
        if (xmin < self.margin_threshold) or (ymin < self.margin_threshold) or (
                xmax > (self.input_shape[1] - self.margin_threshold)) or (
                ymax > (self.input_shape[0] - self.margin_threshold)):
            return conf * self.margin_weight
        else:
            return conf

    def predict(self, split_img_path=None, zoom_ratio=None):
        inputs = []
        images = []
        filenames = []
        file_path = split_img_path
        ratio = zoom_ratio

        if file_path is None:
            file_path = self.split_img_path

        if ratio is None:
            ratio = self.zoom_ratio

        # print('## split file path : ', file_path)

        filelist = glob.glob(file_path + '/*.*')
        filelist.sort()

        # Loading divided images for predication
        inputs, images, filenames = self._get_img_list_by_ch(filelist, self.input_shape[2])

        # run to prediction
        preds = self.model.predict(inputs, batch_size=16, verbose=1)
        results = self.bbox_util.detection_out(preds)

        # make list of obect coordinate (label, xmin, ymin, xmax, ymax)
        rtn_values = []

        for idx, img in enumerate(images):
            # Parse the outputs.
            if len(results[idx]):
                det_label = results[idx][:, 0]
                det_conf = results[idx][:, 1]
                det_xmin = results[idx][:, 2]
                det_ymin = results[idx][:, 3]
                det_xmax = results[idx][:, 4]
                det_ymax = results[idx][:, 5]

                # Get detections with confidence higher than 0.3
                top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]

                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]

                for j in range(top_conf.shape[0]):
                    xmin = int(round(top_xmin[j] * img.shape[1]))
                    ymin = int(round(top_ymin[j] * img.shape[0]))
                    xmax = int(round(top_xmax[j] * img.shape[1]))
                    ymax = int(round(top_ymax[j] * img.shape[0]))
                    conf_val = top_conf[j]
                    label = self.classes[int(top_label_indices[j]) - 1]

                    conf_val = self._realign_weight_of_marginal_box(conf_val, xmin, ymin, xmax, ymax)

                    tmp_obj = PointObject(class_nm=label, confidence=conf_val,
                                          zoom_ratio=ratio)

                    # set absolute coordinate
                    tmp_obj.setAbsoluteCoord([xmin, ymin, xmax, ymax], filenames[idx])

                    # tmp_obj.setCropImgCoord(ori_img)

                    rtn_values.append(tmp_obj)

        return rtn_values

    def predict_from_obj_list(self, img_obj_list, img_coord_list, zoom_ratio=None, return_timeinfo=False, conf_thres=0.5):
        """
        predict from obj list
        """

        inputs = []
        images = []
        ratio = zoom_ratio

        if ratio is None:
            ratio = self.zoom_ratio

        # Loading divided images for predication
        inputs, images = self._get_img_list_by_ch_obj(img_obj_list, self.input_shape[2])

        # self.model = multi_gpu_model(self.model, gpus=2)

        # run to prediction
        start_time = time.process_time()
        preds = self.model.predict(inputs, batch_size=self.batch_size, verbose=1)
        end_time = time.process_time()
        pure_all_predict_time = end_time - start_time

        results = self.bbox_util.detection_out(preds, keep_top_k=200, confidence_threshold=conf_thres)
        #results = self.bbox_util.detection_out2(preds, images, keep_top_k=300, NMS_ALGORITHM=self.NMS_ALGORITHM)

        # make list of obect coordinate (label, xmin, ymin, xmax, ymax)
        rtn_values = []

        for idx, img in enumerate(images):
            # Parse the outputs.
            if len(results[idx]):
                det_label = results[idx][:, 0]
                det_conf = results[idx][:, 1]
                det_xmin = results[idx][:, 2]
                det_ymin = results[idx][:, 3]
                det_xmax = results[idx][:, 4]
                det_ymax = results[idx][:, 5]

                # Get detections with confidence higher than 0.3
                top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]

                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]

                for j in range(top_conf.shape[0]):
                    xmin = int(round(top_xmin[j] * img.shape[1]))
                    ymin = int(round(top_ymin[j] * img.shape[0]))
                    xmax = int(round(top_xmax[j] * img.shape[1]))
                    ymax = int(round(top_ymax[j] * img.shape[0]))
                    conf_val = top_conf[j]
                    label = self.classes[int(top_label_indices[j]) - 1]
                    zoomratio = img_coord_list[idx][4]

                    conf_val = self._realign_weight_of_marginal_box(conf_val, xmin, ymin, xmax, ymax)

                    tmp_obj = PointObject(class_nm=label, confidence=conf_val, zoom_ratio=zoomratio)

                    # set absolute coordinate
                    tmp_obj.setAbsoluteCoordByObj([xmin, ymin, xmax, ymax], img_coord_list[idx])

                    rtn_values.append(tmp_obj)

        time_info = {
            "pure_all_predict_time": pure_all_predict_time
        }

        if return_timeinfo:
            return rtn_values, time_info
        else:
            return rtn_values

    def predict_single_nongray_img_input(self, img, confidence_threshold=0.3, is_bgr=True, return_format="po"):
        """
        predict a single image that is already has model input size and return the point object list.
        assumes that the input image is converted to gray scale when given.
        """
        img_h, img_w, _ = img.shape

        assert img_h == self.input_shape[1]
        assert img_w == self.input_shape[0]

        avail_return_format = ["po", "pylist"]

        assert return_format in avail_return_format

        # mif_imgmat = self._convert_rawimgmat_to_modelinputformat(img, is_bgr = is_bgr)

        mif_imgmat = img

        dl.debug(mif_imgmat)

        inputs = [mif_imgmat]

        inputs = np.array(inputs)

        preds = self.model.predict(inputs, batch_size=1, verbose=1)

        results = self.bbox_util.detection_out(preds)

        # convert results to point object list

        output_list = []

        result_for_first_batch = results[0]

        for result in result_for_first_batch:

            conf = result[1]

            if conf < confidence_threshold:
                continue

            class_index = result[0]

            class_nm = self.classes[int(class_index) - 1]

            if return_format == "po":

                po = PointObject(class_nm=class_nm, confidence=conf)
                coord = result[2:6]

                coord = coord.tolist()
                po.set_absolute_coord(coord)

                output_list.append(po)

            elif return_format == "pylist":
                pylist = [class_nm, conf]
                coord = result[2:6]
                pylist = pylist + coord.tolist()

                output_list.append(pylist)

        return output_list

    # 이미지 채널에 따른 파일 리스트 분류
    def _get_img_list_by_ch(self, filelist, channel):
        """
        load images from filepaths and preprocess images based on the desired channel size
        """

        inputs = []
        images = []
        filenames = []

        for filename in filelist:

            # load to split image with grayscale
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            images.append(img.copy())

            if channel == 1:

                img = img.reshape(img.shape[0], img.shape[1], 1)
                img = img / 255.0

                filenames.append(filename)
                inputs.append(img.copy())

            else:
                ret = np.empty((img.shape[0], img.shape[1], 3), dtype=np.float32)
                ret[:, :, 0] = img
                ret[:, :, 1] = ret[:, :, 2] = ret[:, :, 0]
                img = ret

                filenames.append(filename)
                inputs.append(img.copy())

        if self.input_shape[2] == 1:
            inputs = np.array(inputs)
        else:
            inputs = preprocess_input(np.array(inputs))

        return inputs, images, filenames

    # 이미지 채널에 따른 파일 리스트 분류
    def _get_img_list_by_ch_obj(self, img_obj_list, channel):
        """
        preprocess image data based on desired output image channel size

        by default all images will be converted to grayscale and then processed further.
        """

        inputs = []
        images = []

        for img in img_obj_list:

            # load to split image with grayscale
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            images.append(img.copy())

            if channel == 1:

                img = img.reshape(img.shape[0], img.shape[1], 1)
                img = img / 255.0

                inputs.append(img.copy())

            else:
                ret = np.empty((img.shape[0], img.shape[1], 3), dtype=np.float32)
                ret[:, :, 0] = img
                ret[:, :, 1] = ret[:, :, 2] = ret[:, :, 0]
                img = ret

                inputs.append(img.copy())

        if self.input_shape[2] == 1:
            inputs = np.array(inputs)
        else:
            inputs = preprocess_input(np.array(inputs))

        return inputs, images

    def _convert_rawimgmat_to_modelinputformat(self, rawimgmat, is_bgr=True):
        """
        convert three color img matrix to normalized gray 3-channel img matrix
        """

        assert tuple(self.input_shape) == rawimgmat.shape, "model input shape: {}, given img shape: {}".format(
            self.input_shape, rawimgmat.shape)

        if is_bgr:
            gray_imgmat = cv2.cvtColor(rawimgmat, cv2.COLOR_BGR2GRAY)
        else:
            # assume it is rgb format
            gray_imgmat = cv2.cvtColor(rawimgmat, cv2.COLOR_RGB2GRAY)

        ch3_gray_imgmat = np.empty(shape=self.input_shape, dtype=np.float32)
        ch3_gray_imgmat[:, :, 0] = gray_imgmat
        ch3_gray_imgmat[:, :, 1] = gray_imgmat
        ch3_gray_imgmat[:, :, 2] = gray_imgmat

        # normalize the values to 0~1

        ch3_gray_imgmat = ch3_gray_imgmat / 255.0

        return ch3_gray_imgmat

    def _convert_rawimgmat_to_255normalized(self, rawimgmat):

        return rawimgmat / 255.0







