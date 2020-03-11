import cv2
import numpy as np

img = cv2.imread('/home/aicr/cuongnd/aicr.core/classifier_CRNN/form/IDCARD.jpg',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
cv2.imwrite('/form/IDCARD.jpg', erosion)
cv2.imshow('result',erosion)
cv2.waitKey(0)