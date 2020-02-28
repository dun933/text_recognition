import cv2, os
import numpy as np
from matplotlib import pyplot as plt
import time

template_dir='template'
template_imgs=['field1.jpg','field3.jpg','field4.jpg']
#template_imgs=['field1_0.5.jpg','field3_0.5.jpg','field4_0.5.jpg']


origin_img=os.path.join(template_dir,'0001_ori.jpg')
origin_pts = np.float32([[110, 260], [1966, 2376], [149, 3350]])
#origin_pts = np.float32([[55, 130], [983, 1188], [74, 1675]])

bboxes=[[24,94,708,684],
        [1700,2104,736,636],
        [14,3098,532,390]]

def find_loc(input_img, template, thres = 0.7, method='cv2.TM_CCORR_NORMED'):
    res = cv2.matchTemplate(input_img, template, 3)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    pos= None
    if max_val>thres:
        pos=max_loc
    return pos[0],pos[1]

def calib_image(target_img, target_path='', debug=False, fast=True): #target_img is cv2 image
    gray_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    list_pts=[]
    if fast:
        for idx, img in enumerate(template_imgs):
            left = bboxes[idx][0]
            top = bboxes[idx][1]
            right = bboxes[idx][0] + bboxes[idx][2]
            bottom = bboxes[idx][1] + bboxes[idx][3]
            crop_img = gray_img[top:bottom, left:right]
            template = cv2.imread(os.path.join(template_dir, img), 0)
            locx, locy = find_loc(crop_img, template)
            locx, locy = locx + left, locy + top
            list_pts.append((locx, locy))
    else:
        for img in template_imgs:
            template = cv2.imread(os.path.join(template_dir, img), 0)
            locx, locy = find_loc(gray_img, template)
            list_pts.append((locx, locy))

    target_pts=np.asarray(list_pts, dtype=np.float32)
    affine_trans = cv2.getAffineTransform(target_pts, origin_pts)
    trans_img = cv2.warpAffine(target_img, affine_trans, (target_img.shape[1], target_img.shape[0]))
    if debug:
        print (target_pts)
        cv2.imwrite(target_path.replace('.jpg','_transform.jpg'), trans_img)
    return trans_img

def crop_image(input_img, bbox=[905,1010,1300,138]):
    print('crop')
    offset_x = 0
    offset_y = 0
    crop_img = input_img[bbox[1]+offset_y:bbox[1] + bbox[3]+offset_y, bbox[0]+offset_x:bbox[0] + bbox[2]+offset_x]
    return crop_img

def background_subtract(image):
    bgr_path='C:/Users/nd.cuong1/Downloads/Template_Matching-master/data/test_tm/field1.jpg'
    background=cv2.imread(bgr_path, 0)
    result = cv2.subtract(background, image)
    result_inv = cv2.bitwise_not(result)
    #cv2.imshow('result',result_inv)
    #cv2.waitKey(0)
    return result_inv


if __name__ == "__main__":
    target_path='/home/aicr/cuongnd/text_recognition/crnn_pbcquoc/template/0001_tungnt.jpg'
    target_img=cv2.imread(target_path)
    #target_img=cv2.resize(target_img,(1240, 1754))
    begin=time.time()
    calib_image(target_img,target_path, debug=True)
    end=time.time()
    print('Time:',end-begin,'seconds')
