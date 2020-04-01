import time, os
import cv2, math
import numpy as np

RADIAN_PER_DEGREE = 0.0174532

class Template_info:
    def __init__(self, name, field_dir, field_imgs, field_locs, field_rois=None,
                 scales=(0.6, 1.2, 0.2), rotations=(-10, 10, 5), confidence=0.7, debug=False):
        self.name = name
        self.field_dir = field_dir
        self.confidence = confidence
        self.field_imgs = field_imgs
        self.field_locs = field_locs
        self.field_rois = field_rois
        self.debug=debug
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

        num_scales = int((scales[1] - scales[0]) / scales[2] + 2)
        num_rotations = int((rotations[1] - rotations[0]) / rotations[2] + 1)
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
        for scale in list_scales:
            for rotation in list_rotations:
                print('scale', scale, ', rotation', rotation)
                temp = dict()
                temp['scale'] = scale
                temp['rotation'] = rotation
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
                temp['data'] = crop_rotated
                field['list_samples'].append(temp)
                if self.debug:
                    cv2.imshow('result', rotated)
                    cv2.imshow('result_crop', crop_rotated)
                    ch = cv2.waitKey(0)
                    if ch == 27:
                        break

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
                          template_dir + '/template_VIB',
                          ['field1.jpg', 'field3.jpg', 'field4.jpg'],
                          np.float32([[259, 331], [2125.5, 2424], [230, 3318.5]]),
                          [[24, 94, 708, 684],[1700, 2104, 736, 636], [14, 3098, 532, 390]])
        kk = 1

    def add_template(self, template_name, field_dir, field_imgs, field_locs, field_rois, scales=(0.6, 1.2, 0.2),
                     rotations=(-10, 10, 5), debug=False):
        temp = Template_info(template_name, field_dir, field_imgs, field_locs, field_rois, scales, rotations, debug)
        self.template_list.append(temp)


    def find_field(self, input_img, template, thres=0.7, method='cv2.TM_CCORR_NORMED'):
        res = cv2.matchTemplate(input_img, template, 3)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        pos = None
        if max_val > thres:
            pos = max_loc
        return max_val, pos[0] + template.shape[1] / 2, pos[1] + template.shape[0] / 2

    def calib_template(self, template_name, target_img, target_path='', debug=True,
                       fast=False):  # target_img is cv2 image

        for template in self.template_list:
            if template.name==template_name:
                template_data=template
                break

        gray_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        list_pts = []
        if fast:
            # for idx, img in enumerate(template_data.list_field_samples):
            #     left = bboxes[idx][0]
            #     top = bboxes[idx][1]
            #     right = bboxes[idx][0] + bboxes[idx][2]
            #     bottom = bboxes[idx][1] + bboxes[idx][3]
            #     crop_img = gray_img[top:bottom, left:right]
            #     template = cv2.imread(os.path.join(template_dir, img), 0)
            #     conf, locx, locy = self.find_field(crop_img, template)
            #     locx, locy = locx + left, locy + top
            #     list_pts.append((locx, locy))
            kk=1
        else:
            for idx, field in enumerate(template_data.list_field_samples):
                print(field['name'])
                max_conf=0
                final_locx, final_locy=-1,-1
                scale, rotation=1.0,0
                for sample in field['list_samples']:
                    sample_data = sample['data']
                    conf, locx, locy = self.find_field(gray_img, sample_data)
                    if conf>max_conf:
                        max_conf=conf
                        final_locx, final_locy =locx, locy
                        scale, rotation =sample['scale'], sample['rotation']
                print(scale, rotation)
                list_pts.append((final_locx, final_locy))

        target_pts = np.asarray(list_pts, dtype=np.float32)
        affine_trans = cv2.getAffineTransform(target_pts, template_data.field_locs)
        trans_img = cv2.warpAffine(target_img, affine_trans, (target_img.shape[1], target_img.shape[0]))
        if debug:
            print(target_pts)
            trans_img=cv2.resize(trans_img,(int(trans_img.shape[1]/4),int(trans_img.shape[0]/4)))
            cv2.imshow('transform',trans_img)
            cv2.waitKey(0)
            #cv2.imwrite(target_path.replace('.jpg', '_transform.jpg'), trans_img)
        return trans_img

    def crop_image(input_img, bbox=[905, 1010, 1300, 138]):
        print('crop')
        offset_x = 0
        offset_y = 0
        crop_img = input_img[bbox[1] + offset_y:bbox[1] + bbox[3] + offset_y,
                   bbox[0] + offset_x:bbox[0] + bbox[2] + offset_x]
        return crop_img

    def background_subtract(image,
                            bgr_path='C:/Users/nd.cuong1/Downloads/Template_Matching-master/data/test_tm/field1.jpg'):
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # background=cv2.imread(bgr_path, 0)
        background = cv2.imread(bgr_path)
        result = cv2.subtract(background, image)
        result_inv = cv2.bitwise_not(result)
        cv2.imshow('result', result_inv)
        cv2.waitKey(0)
        return result_inv


if __name__ == "__main__":
    # target_path = '../form/template_VIB/0001_ori.jpg'
    # target_img = cv2.imread(target_path)
    # # target_img=cv2.resize(target_img,(1240, 1754))
    # begin = time.time()
    # calib_image(target_img, target_path, debug=True)
    # end = time.time()

    target_img=cv2.imread('../form/template_VIB/0001_ori.jpg')

    kk = MatchingTemplate()
    kk.calib_template('VIB_form',target_img)

    # print('Time:', end - begin, 'seconds')
