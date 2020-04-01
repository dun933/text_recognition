import time, os
import cv2, math
import numpy as np

RADIAN_PER_DEGREE = 0.0174532


class Template_info:
    def __init__(self, name, template_path, field_dir, field_imgs, field_locs, field_rois=None,
                 scales=(0.6, 1.2, 0.2), rotations=(-10, 10, 5), confidence=0.7, debug=False):
        self.name = name
        self.template_img = cv2.imread(template_path,0)
        self.template_width = self.template_img.shape[1]
        self.template_height = self.template_img.shape[0]
        self.field_dir = field_dir
        self.confidence = confidence
        self.field_imgs = field_imgs
        self.field_locs = field_locs
        self.field_rois = field_rois
        self.debug = debug
        self.list_field_samples = []
        for idx, img in enumerate(self.field_imgs):
            field = dict()
            field['dir'] = field_dir
            field['name'] = img
            field['loc'] = field_locs[idx]
            field['roi'] = None
            if field_rois is not None:
                field['roi'] = field_rois[idx]

            self.createSamples(field, scales, rotations)
            self.list_field_samples.append(field)

    def createSamples(self, field, scales, rotations):
        print('add_template', field['name'])
        list_scales = []
        list_rotations = []

        num_scales = round((scales[1] - scales[0]) / scales[2]) + 1
        num_rotations = round((rotations[1] - rotations[0]) / rotations[2]) + 1
        for i in range(num_scales):
            list_scales.append(round(scales[0] + i * scales[2], 4))
        for i in range(num_rotations):
            list_rotations.append(round(rotations[0] + i * rotations[2], 4))

        field['list_samples'] = []
        template_data = cv2.imread(os.path.join(field['dir'], field['name']), 0)
        w = template_data.shape[1]
        h = template_data.shape[0]
        bgr_val = int((int(template_data[0][0]) + int(template_data[0][w - 1]) + int(
            template_data[h - 1][w - 1]) + int(template_data[h - 1][0])) / 4)
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

            (h, w) = template_data.shape[:2]
            (cX, cY) = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D((cX, cY), -rotation, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            rotated = cv2.warpAffine(template_data, M, (nW, nH), borderValue=bgr_val)

            # (h_rot, w_rot) = rotated.shape[:2]
            # (cX_rot, cY_rot) = (w_rot // 2, h_rot // 2)
            #
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
            print('origin size', crop_w, crop_h)

            for scale in list_scales:
                temp = dict()
                temp['rotation'] = rotation
                temp['scale'] = scale
                print('scale', scale, ', rotation', rotation)
                crop_rotate_resize = cv2.resize(crop_rotated, (int(scale * crop_w), int(scale * crop_h)))
                print('resize size', int(scale * crop_w), int(scale * crop_h))
                temp['data'] = crop_rotate_resize
                if self.debug:
                    cv2.imshow('result', crop_rotated)
                    cv2.imshow('result_crop', crop_rotate_resize)
                    ch = cv2.waitKey(0)
                    if ch == 27:
                        break
                field['list_samples'].append(temp)


class MatchingTemplate:
    def __init__(self):
        self.template_dir = ''
        self.template_names = []
        self.template_list = []
        self.initTemplate()

    def initTemplate(self, template_dir='../form', list_template_name=[]):
        # self.add_template('CMND_old',
        #                   template_dir + '/template_IDcard',
        #                   ['template_1.jpg', 'template_2.jpg', 'template_3.jpg', 'template_4.jpg'],
        #                   np.float32([[110, 260], [110, 260], [1979, 2397], [157, 3234]]),
        #                   None)
        self.add_template('VIB_form',
                          template_dir +'/template_VIB/0001_ori.jpg',
                          template_dir + '/template_VIB',
                          ['field1.jpg', 'field3.jpg', 'field4.jpg'],
                          [[258.5, 330.5], [2125.0, 2423.5], [229.5, 3318.0]],
                          [[24, 94, 708, 684], [1700, 2104, 736, 636], [14, 3098, 532, 390]])

    def add_template(self, template_name, template_path, field_dir, field_imgs, field_locs, field_rois, scales=(0.9, 1.1, 0.1),
                     rotations=(-2, 2, 2), debug=True):
        temp = Template_info(template_name, template_path, field_dir, field_imgs, field_locs, field_rois, scales, rotations, debug)
        self.template_list.append(temp)

    def find_field(self, input_img, template, thres=0.7, method='cv2.TM_CCORR_NORMED'):
        res = cv2.matchTemplate(input_img, template, 3)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        pos = None
        if max_val > thres:
            pos = max_loc
        return max_val, pos[0] + template.shape[1] / 2, pos[1] + template.shape[0] / 2

    def calib_template(self, template_name, target_img, target_path='', debug=False,
                       fast=True):  # target_img is cv2 image

        for template in self.template_list:
            if template.name == template_name:
                template_data = template
                break

        gray_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        list_pts = []
        if fast:
            for idx, field in enumerate(template_data.list_field_samples):
                print(field['name'])
                max_conf = 0
                final_locx, final_locy = -1, -1
                scale, rotation = 1.0, 0

                left = field['roi'][0]
                top = field['roi'][1]
                right = field['roi'][0] + field['roi'][2]
                bottom = field['roi'][1] + field['roi'][3]
                crop_img = gray_img[top:bottom, left:right]

                for sample in field['list_samples']:
                    sample_data = sample['data']
                    conf, locx, locy = self.find_field(crop_img, sample_data)
                    print(conf, sample['scale'], sample['rotation'])
                    if conf > max_conf:
                        max_conf = conf
                        final_locx, final_locy = locx, locy
                        scale, rotation = sample['scale'], sample['rotation']
                print(scale, rotation)
                final_locx, final_locy = final_locx + left, final_locy + top
                print(final_locx, final_locy)
                list_pts.append((final_locx, final_locy))
        else:
            for idx, field in enumerate(template_data.list_field_samples):
                print(field['name'])
                max_conf = 0
                final_locx, final_locy = -1, -1
                scale, rotation = 1.0, 0
                for sample in field['list_samples']:
                    sample_data = sample['data']
                    conf, locx, locy = self.find_field(gray_img, sample_data)
                    print(conf, sample['scale'], sample['rotation'])
                    if conf > max_conf:
                        max_conf = conf
                        final_locx, final_locy = locx, locy
                        scale, rotation = sample['scale'], sample['rotation']
                print(scale, rotation)
                print(final_locx, final_locy)
                list_pts.append((final_locx, final_locy))

        img_pts = np.asarray(list_pts, dtype=np.float32)
        ori_pts = np.asarray(template_data.field_locs, dtype=np.float32)
        if len(img_pts) == 3:  # affine transformation
            affine_trans = cv2.getAffineTransform(img_pts, ori_pts)
            trans_img = cv2.warpAffine(target_img, affine_trans, (template_data.template_width, template_data.template_height))
        if len(img_pts) == 4:  # perspective transformation
            perspective_trans = cv2.getPerspectiveTransform(img_pts, ori_pts)
            trans_img = cv2.warpPerspective(target_img, perspective_trans, (template_data.template_width, template_data.template_height))
        if debug:
            print(img_pts)
            trans_img = cv2.resize(trans_img, (int(trans_img.shape[1] / 4), int(trans_img.shape[0] / 4)))
            cv2.imshow('transform', trans_img)
            cv2.waitKey(0)
            # cv2.imwrite(target_path.replace('.jpg', '_transform.jpg'), trans_img)
        return trans_img

    def crop_image(self, input_img, bbox=[905, 1010, 1300, 138]):
        print('crop')
        offset_x = 0
        offset_y = 0
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
    target_img = cv2.imread('../form/template_VIB/0001_ori.jpg')

    kk = MatchingTemplate()

    begin = time.time()
    kk.calib_template('VIB_form', target_img)
    end = time.time()

    print('Time:', end - begin, 'seconds')
