import cv2, os
import numpy as np



def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['png', 'jpg', 'JPG', 'jpeg', 'PNG']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def resizeCMND_old(img_dir, img_name, maxW=1000, maxH=631):
    print (img_name)
    img= cv2.imread(os.path.join(img_dir,img_name))
    CMND_ratio=maxW/maxH
    imgW=img.shape[1]
    imgH=img.shape[0]
    img_ratio=imgW/imgH

    if img_ratio<CMND_ratio:
        newH=maxH
        newW=int(newH*img_ratio)

    else:
        newW=maxW
        newH=int(newW/img_ratio)
    resize=cv2.resize(img,(newW,newH))
    cv2.imwrite(os.path.join('/home/aicr/cuongnd/aicr.core/data/IDcard/CMND_old_1/resize',img_name.replace('.jpeg','.jpg').replace('.png','.jpg').replace('.PNG','.jpg')),resize)

def erode(img_path):
    img = cv2.imread(img_path)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite('/home/aicr/cuongnd/aicr.core/test/bgr_tax_code_2.jpg', erosion)
    cv2.imshow('result', erosion)
    cv2.waitKey(0)


# !/usr/bin/env python

import cv2
import numpy as np

def warp_homography():
    im_src = cv2.imread('cmnd2.jpg')
    #pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])
    #pts_src = np.array([[184, 220], [403, 216], [428, 298], [172, 450]])
    pts_src = np.array([[223, 168], [397, 151], [731, 129], [378, 367]])
    pts_src = np.array([[222, 224], [494, 90], [786, 172], [378, 367]])
    cv2.line(im_src,(pts_src[0][0],pts_src[0][1]),(pts_src[2][0],pts_src[2][1]), color=(0,0,255), thickness =2)
    cv2.line(im_src,(pts_src[1][0],pts_src[1][1]),(pts_src[3][0],pts_src[3][1]), color=(0,0,255), thickness =2)

    # Read destination image.
    im_dst = cv2.imread('cmnd1.jpg')
    #pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])
    #pts_dst = np.array([[308, 311], [452, 379], [434, 435], [200, 429]])
    pts_dst = np.array([[141, 111], [343, 93], [737, 66], [324, 343]])
    pts_dst = np.array([[140, 176], [454, 21], [802, 117], [324, 343]])
    cv2.line(im_dst,(pts_dst[0][0],pts_dst[0][1]),(pts_dst[2][0],pts_dst[2][1]), color=(0,0,255), thickness =2)
    cv2.line(im_dst,(pts_dst[1][0],pts_dst[1][1]),(pts_dst[3][0],pts_dst[3][1]), color=(0,0,255), thickness =2)

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    #cv2.getPerspectiveTransform()

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
    cv2.imwrite('result_2.jpg',im_out)

    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)

    cv2.waitKey(0)


if __name__ == "__main__":
    # data_dir= '../../data/IDcard/CMND_old_1'
    # img_list=get_list_file_in_folder(data_dir)
    #
    # for img in img_list:
    #     resizeCMND_old(data_dir,img)
    warp_homography()
    #erode('/home/aicr/cuongnd/aicr.core/test/bgr_tax_code.jpg')
