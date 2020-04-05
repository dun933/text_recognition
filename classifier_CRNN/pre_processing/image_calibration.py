import time, os
import cv2, math
import numpy as np

RADIAN_PER_DEGREE = 0.0174532
debug = False


def draw_bboxes(img, bboxes):
    # e.g: bboxes= [(0,0),(0,5),(5,5),(5,0)]
    if len(img.shape) != 3:
        img_RGB = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_RGB = img
    color_red = (0, 0, 255)
    for bbox in bboxes:
        cv2.line(img_RGB, bbox[0], bbox[1], color=color_red, thickness=2)
        cv2.line(img_RGB, bbox[1], bbox[2], color=color_red, thickness=2)
        cv2.line(img_RGB, bbox[2], bbox[3], color=color_red, thickness=2)
        cv2.line(img_RGB, bbox[3], bbox[0], color=color_red, thickness=2)

    img_RGB = cv2.resize(img_RGB, (int(img_RGB.shape[1] / 3), int(img_RGB.shape[0] / 3)))
    cv2.imshow('draw bboxes', img_RGB)
    cv2.waitKey(0)


class Template_info:
    def __init__(self, name, template_path, field_bboxes, field_rois_extend=1.0, field_search_areas=None,
                 confidence=0.7, scales=(0.9, 1.1, 0.1), rotations=(-2, 2, 2), normalize_width=1654):
        self.name = name
        self.template_img = cv2.imread(template_path, 0)
        self.resize_ratio, self.template_img = self.resize_normalize(self.template_img, normalize_width)
        self.template_width = self.template_img.shape[1]
        self.template_height = self.template_img.shape[0]
        self.confidence = confidence
        self.field_bboxes = field_bboxes
        self.field_rois_extend = field_rois_extend
        self.field_search_areas = field_search_areas
        self.field_locs = []
        self.list_field_samples = []
        for idx, bbox in enumerate(self.field_bboxes):
            bbox = self.resize_bbox(bbox, self.resize_ratio)

            field = dict()
            field['name'] = str(idx)
            field['loc'] = (bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2)
            self.field_locs.append(field['loc'])
            field['search_area'] = None
            if field_search_areas is not None:
                field['search_area'] = self.resize_bbox(field_search_areas[idx], self.resize_ratio)
            else:
                field['data'] = self.crop_image(self.template_img, bbox)
                # cv2.imwrite(field['name']+'.jpg', field['data'])
                # cv2.waitKey(0)
                field_w = max(field['data'].shape[1], 80)
                field_h = max(field['data'].shape[0], 80)
                extend_x = int(field_rois_extend * field_w)
                extend_y = int(field_rois_extend * field_h)
                left = max(int(field['loc'][0] - field_w / 2 - extend_x), 0)
                top = max(int(field['loc'][1] - field_h / 2 - extend_y), 0)
                right = min(int(field['loc'][0] + field_w / 2 + extend_x), self.template_width)
                bottom = min(int(field['loc'][1] + field_h / 2 + extend_y), self.template_height)
                width = right - left
                height = bottom - top
                field['search_area'] = [left, top, width, height]

            self.createSamples(field, scales, rotations)
            self.list_field_samples.append(field)

    def resize_normalize(self, img, normalize_width=1654):
        w = img.shape[1]
        h = img.shape[0]
        resize_ratio = normalize_width / w
        normalize_height = round(h * resize_ratio)
        resize_img = cv2.resize(img, (normalize_width, normalize_height), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow('resize img', resize_img)
        # cv2.waitKey(0)
        return resize_ratio, resize_img

    def resize_bbox(self, bbox, resize_ratio):
        for i in range(len(bbox)):
            bbox[i] = round(bbox[i] * resize_ratio)
        return bbox

    def crop_image(self, input_img, bbox, offset_x=0, offset_y=0):
        print('crop')
        crop_img = input_img[bbox[1] + offset_y:bbox[1] + bbox[3] + offset_y,
                   bbox[0] + offset_x:bbox[0] + bbox[2] + offset_x]
        return crop_img

    def createSamples(self, field, scales, rotations):
        print('Add_template', field['name'])
        list_scales = []
        list_rotations = []

        num_scales = round((scales[1] - scales[0]) / scales[2]) + 1
        num_rotations = round((rotations[1] - rotations[0]) / rotations[2]) + 1
        for i in range(num_scales):
            list_scales.append(round(scales[0] + i * scales[2], 4))
        for i in range(num_rotations):
            list_rotations.append(round(rotations[0] + i * rotations[2], 4))

        field['list_samples'] = []
        field_data = field['data']
        w = field_data.shape[1]
        h = field_data.shape[0]
        bgr_val = int((int(field_data[0][0]) + int(field_data[0][w - 1]) + int(
            field_data[h - 1][w - 1]) + int(field_data[h - 1][0])) / 4)
        for rotation in list_rotations:
            abs_rotation = abs(rotation)
            if (w < h):
                if (abs_rotation <= 45):
                    sa = math.sin(abs_rotation * RADIAN_PER_DEGREE)
                    ca = math.cos(abs_rotation * RADIAN_PER_DEGREE)
                    newHeight = (int)((h - w * sa) / ca)
                    # newHeight = newHeight - ((h - newHeight) % 2)
                    szOutput = (w, newHeight)
                else:
                    sa = math.sin((90 - abs_rotation) * RADIAN_PER_DEGREE)
                    ca = math.cos((90 - abs_rotation) * RADIAN_PER_DEGREE)
                    newWidth = (int)((h - w * sa) / ca)
                    # newWidth = newWidth - ((w - newWidth) % 2)
                    szOutput = (newWidth, w)
            else:
                if (abs_rotation <= 45):
                    sa = math.sin(abs_rotation * RADIAN_PER_DEGREE)
                    ca = math.cos(abs_rotation * RADIAN_PER_DEGREE)
                    newWidth = (int)((w - h * sa) / ca)
                    # newWidth = newWidth - ((w - newWidth) % 2)
                    szOutput = (newWidth, h)
                else:
                    sa = math.sin((90 - rotation) * RADIAN_PER_DEGREE)
                    ca = math.cos((90 - rotation) * RADIAN_PER_DEGREE)
                    newHeight = (int)((w - h * sa) / ca)
                    # newHeight = newHeight - ((h - newHeight) % 2)
                    szOutput = (h, newHeight)

            (h, w) = field_data.shape[:2]
            (cX, cY) = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D((cX, cY), -rotation, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            rotated = cv2.warpAffine(field_data, M, (nW, nH), borderValue=bgr_val)

            # (h_rot, w_rot) = rotated.shape[:2]
            # (cX_rot, cY_rot) = (w_rot // 2, h_rot // 2)
            # pt1=(int(cX_rot-3), int(cY_rot-3))
            # pt2=(int(cX_rot+3), int(cY_rot+3))
            # pt3=(int(cX_rot-3), int(cY_rot+3))
            # pt4=(int(cX_rot+3), int(cY_rot-3))
            # cv2.line(rotated,pt1,pt2,color=255)
            # cv2.line(rotated,pt3,pt4,color=255)

            offset_X = int((nW - szOutput[0]) / 2)
            offset_Y = int((nH - szOutput[1]) / 2)

            crop_rotated = rotated[offset_Y:nH - offset_Y - 1, offset_X:nW - offset_X - 1]
            crop_w = crop_rotated.shape[1]
            crop_h = crop_rotated.shape[0]
            # rint('origin size', crop_w, crop_h)

            for scale in list_scales:
                temp = dict()
                temp['rotation'] = rotation
                temp['scale'] = scale
                print('scale', scale, ', rotation', rotation)
                crop_rotate_resize = cv2.resize(crop_rotated, (int(scale * crop_w), int(scale * crop_h)))
                # print('resize size', int(scale * crop_w), int(scale * crop_h))
                temp['data'] = crop_rotate_resize
                if debug:
                    cv2.imshow('result', crop_rotated)
                    cv2.imshow('result_crop', crop_rotate_resize)
                    ch = cv2.waitKey(0)
                    if ch == 27:
                        cv2.imwrite('result.jpg', crop_rotated)
                        break
                field['list_samples'].append(temp)

    def draw_template(self):
        list_bboxes =[]
        for bbox in self.field_bboxes:
            left = bbox[0]
            top = bbox[1]
            right = bbox[0] + bbox[2]
            bottom = bbox[1] + bbox[3]
            bboxes = [(left, top), (right, top), (right, bottom), (left, bottom)]
            list_bboxes.append(bboxes)
        draw_bboxes(self.template_img, list_bboxes)


class MatchingTemplate:
    def __init__(self):
        self.template_dir = ''
        self.template_names = []
        self.template_list = []
        self.initTemplate()
        self.matching_results = []

    def initTemplate(self, template_dir='../form', list_template_name=[]):
        self.add_template(template_name='VIB_form',
                          template_path=template_dir + '/template_VIB/0001_ori.jpg',
                          field_bboxes=[[96, 157, 222, 108], [1320, 1595, 192, 42], [96, 2170, 98, 104]],
                          field_rois_extend=1.0,
                          field_search_areas=None,
                          confidence=0.7,
                          scales=(0.9, 1.1, 0.1),
                          rotations=(-2, 2, 2))

        # self.add_template(template_name='CMND_old',
        #                   template_path=template_dir + '/template_IDcard/2_ori.jpg',
        #                   field_dir=template_dir + '/template_IDcard',
        #                   field_imgs=['template_1.jpg', 'template_2.jpg', 'template_3.jpg', 'template_4.jpg'],
        #                   field_locs=[[141.0, 111.5], [343.0, 93.5], [737.0, 66.0], [324.5, 343.0]],
        #                   field_rois=None,
        #                   field_rois_extend=2.0,
        #                   confidence=0.7,
        #                   scales=(0.6, 1.2, 0.2),
        #                   rotations=(-10, 10, 5))

    def add_template(self, template_name, template_path, field_bboxes, field_rois_extend=1.0, field_search_areas=None,
                     confidence=0.7, scales=(0.9, 1.1, 0.1), rotations=(-2, 2, 2)):
        temp = Template_info(template_name, template_path, field_bboxes, field_rois_extend, field_search_areas,
                             confidence, scales, rotations)
        self.template_list.append(temp)

    def clear_template(self):
        self.template_list.clear()

    def check_template(self, template_name):
        template_data = None
        for template in self.template_list:
            if template.name == template_name:
                template_data = template
                break
        if template_data is None:
            print('Cannot find template', template_name, 'in database')
        return template_data

    def draw_template(self, template_name):
        template_data = self.check_template(template_name)
        if template_data is None:
            return
        template_data.draw_template()

    def find_field(self, input_img, field, thres=0.7, fast=True, method='cv2.TM_CCORR_NORMED'):
        max_conf = 0
        final_locx, final_locy = -1, -1
        final_sample = None

        if len(input_img.shape) == 3:  # BGR
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        if fast:
            left = field['search_area'][0]
            top = field['search_area'][1]
            right = field['search_area'][0] + field['search_area'][2]
            bottom = field['search_area'][1] + field['search_area'][3]
            input_img = input_img[top:bottom, left:right]

        for sample in field['list_samples']:
            sample_data = sample['data']
            res = cv2.matchTemplate(input_img, sample_data, 3)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > max_conf and max_val > thres:
                max_conf = max_val
                final_locx, final_locy = max_loc[0] + sample_data.shape[1] / 2, max_loc[1] + sample_data.shape[0] / 2
                final_sample = sample
        if fast:
            final_locx, final_locy = final_locx + field['search_area'][0], final_locy + field['search_area'][1]
        print('Score:', round(max_conf, 4), 'Scale:', final_sample['scale'], 'Angle:', final_sample['rotation'],
              'Location:', final_locx, final_locy)

        # get rec result
        if final_sample is None:
            return 0, -1, -1

        x0 = final_locx
        y0 = final_locy

        x1 = x0 - (final_sample['data'].shape[1] / 2) * final_sample['scale']
        y1 = y0 - (final_sample['data'].shape[0] / 2) * final_sample['scale']
        x2 = x0 + (final_sample['data'].shape[1] / 2) * final_sample['scale']
        y2 = y0 + (final_sample['data'].shape[0] / 2) * final_sample['scale']

        ##
        ca = math.cos(final_sample['rotation'] * RADIAN_PER_DEGREE)
        sa = math.sin(final_sample['rotation'] * RADIAN_PER_DEGREE)
        rx1 = round((x0 + (x1 - x0) * ca - (y1 - y0) * sa))
        ry1 = round((y0 + (x1 - x0) * sa + (y1 - y0) * ca))
        rx2 = round((x0 + (x2 - x0) * ca - (y1 - y0) * sa))
        ry2 = round((y0 + (x2 - x0) * sa + (y1 - y0) * ca))
        rx3 = round((x0 + (x2 - x0) * ca - (y2 - y0) * sa))
        ry3 = round((y0 + (x2 - x0) * sa + (y2 - y0) * ca))
        rx4 = round((x0 + (x1 - x0) * ca - (y2 - y0) * sa))
        ry4 = round((y0 + (x1 - x0) * sa + (y2 - y0) * ca))

        self.matching_results = [(rx1, ry1), (rx2, ry2), (rx3, ry3), (rx4, ry4)]
        draw_bboxes(input_img, [self.matching_results])

        return max_conf, final_locx, final_locy

    def calib_template(self, template_name, src_img, fast=True):  # src_img is cv2 image
        print('\nCalib template', template_name)
        template_data = self.check_template(template_name)
        if template_data is None:
            return

        src_img = cv2.resize(src_img, (template_data.template_width, template_data.template_height))
        gray_img = src_img
        if len(src_img.shape) == 3:  # BGR
            gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        list_pts = []

        for idx, field in enumerate(template_data.list_field_samples):
            print(field['name'])
            conf, loc_x, loc_y = self.find_field(gray_img, field, fast=fast, thres=template_data.confidence)
            list_pts.append((loc_x, loc_y))

        src_pts = np.asarray(list_pts, dtype=np.float32)
        dst_pts = np.asarray(template_data.field_locs, dtype=np.float32)
        if len(src_pts) == 3:  # affine transformation
            affine_trans = cv2.getAffineTransform(src_pts, dst_pts)
            trans_img = cv2.warpAffine(src_img, affine_trans,
                                       (template_data.template_width, template_data.template_height))
        if len(src_pts) > 3:  # perspective transformation
            perspective_trans, status = cv2.findHomography(src_pts, dst_pts)
            w, h = template_data.template_width, template_data.template_height
            trans_img = cv2.warpPerspective(src_img, perspective_trans, (w, h))
        return trans_img

    def crop_image(self, input_img, bbox, offset_x=0, offset_y=0):
        print('crop')
        crop_img = input_img[bbox[1] + offset_y:bbox[1] + bbox[3] + offset_y,
                   bbox[0] + offset_x:bbox[0] + bbox[2] + offset_x]
        return crop_img

    def background_subtract(self, image, bgr_path='field1.jpg'):
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # background=cv2.imread(bgr_path, 0)
        background = cv2.imread(bgr_path)
        result = cv2.subtract(background, image)
        result_inv = cv2.bitwise_not(result)
        cv2.imshow('result', result_inv)
        cv2.waitKey(0)
        return result_inv


if __name__ == "__main__":
    src_img = cv2.imread('../form/template_VIB/0001_ori.jpg')
    src_img = cv2.imread('../../data/VIB_page1/vib_page1-12.jpg')
    begin_init = time.time()
    match = MatchingTemplate()
    end_init = time.time()
    print('Time init:', end_init - begin_init, 'seconds')

    match.draw_template('VIB_form')

    begin = time.time()
    calib_img = match.calib_template('VIB_form', src_img, fast=True)
    end = time.time()
    print('Time:', end - begin, 'seconds')

    debug = True
    if debug:
        src_img = cv2.resize(src_img, (int(src_img.shape[1] / 2), int(src_img.shape[0] / 2)))
        trans_img = cv2.resize(calib_img, (int(calib_img.shape[1] / 2), int(calib_img.shape[0] / 2)))
        cv2.imshow('origin', src_img)
        cv2.imshow('transform', trans_img)
        cv2.waitKey(0)
        # cv2.imwrite(target_path.replace('.jpg', '_transform.jpg'), trans_img)
