import cv2, os
import numpy as np
import imutils
from matplotlib import pyplot as plt

# img = cv2.imread(dst, 0)
# img2 = img.copy()
# _template_ = cv2.imread(src, 0)
#w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

#this matches for logo
#methods = ['cv2.TM_CCORR_NORMED']
#methods = ['cv2.TM_CCOEFF_NORMED']

def doRotate(img, degree, scale=1.0):
    (w, h) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, degree, scale)
    return cv2.warpAffine(img, M, (h, w))

def doMatch(_template_, _dst, scale = 1, angle = 10, meth='cv2.TM_CCORR_NORMED', _show = True, debug = False):
    t_img = cv2.imread(_template_, 0)
    template = imutils.resize(doRotate(t_img, angle), width=int(t_img.shape[1]*scale))
    img = cv2.imread(_dst, 0)
    w, h = template.shape[::-1]
    #img2 = img.copy()#doRotate(img, angle)
    #img = imutils.resize(img2, width = int(img2.shape[1] * scale))
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    if (debug):
        print(_dst, _template_, max_val, scale, angle, meth)
    if _show:
        cv2.rectangle(img,top_left, bottom_right, 255, 2)
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()

    return top_left, bottom_right, max_val

def matchArea(src, dst, _meth = ['cv2.TM_CCORR_NORMED'], f_log = "error.log", show=False, debug=False):
    error_log = open(f_log, "a")
    mx, my = (0, 0)
    mval = 0.0
    msc = 0.0
    mrot = 0.0
    for smeth in _meth:
        for rot in np.linspace(-5, 5, num=3):
            for sc in np.linspace(0.5, 1.0, num=5):
                try:
                    cx, cy, val = doMatch(src, dst, scale=sc, angle=rot, meth=smeth, _show=False)
                    if mval < val:
                        mrot = rot
                        mval = val
                        mx = cx
                        my = cy
                        msc = sc
                        _meth = smeth
                except Exception as e:
                    print("template is {} and image is {}".format(src, dst), file=error_log)
                    print("angle is {} and scale is {}".format(rot, sc), file=error_log)
                    print("used method is {}".format(smeth), file=error_log)
                    print("exception occured! error is {}".format(e), file=error_log)
                    continue
    error_log.close()
    if debug: print(dst, smeth, msc, mrot, mval)
    if (mx[0]>0 and show==True):
        img = cv2.imread(dst, 0)
        #img1 = doRotate(img, mrot)
        #img2 = imutils.resize(img1, width = int(img.shape[1] * msc))
        cv2.rectangle(img, mx, my, 255, 2)
        cv2.imshow("detected image", img)
        cv2.waitKey(0)
    return mx, my, msc

def findCorners(imPath):
    _dir = "data/template"
    templates=["logo.jpg", "area-I.jpg", "area-II.jpg", "area-III.jpg", "area-IV.jpg"]
    img = cv2.imread(imPath)#imutils.resize(cv2.imread(imPath), width=1600)
    for tpl in templates:
        px, py, sc = matchArea(os.path.join(_dir, tpl), imPath)
        '''tx = px
        ty = py
        tx[0] = int(px[0] * sc)
        tx[1] = int(px[1] * sc)
        ty[0] = int(py[0] * sc)
        ty[1] = int(py[1] * sc)'''
        cv2.rectangle(img, px, py, 255, 2)
    cv2.imshow("detected image", img)
    cv2.waitKey(0)
    

#_dir = "D:/source/aicr.vn/aicr.core/data/template"
#I-1.jpg logo.jpg logo-I.jpg logo-II.jpg logo-IIa.jpg logo-III.jpg logo-IIIa.jpg logo-II-60.jpg logo-II-30.jpg
# I-2a.jpg 
#_src = os.path.join(_dir, "I-2.jpg")
_dir = "data/IDcard/CMND_old_1"
listdir = ["1.jpg", "7.jpeg"]
#listdir = os.listdir(_dst)
#listdir.sort()
for _f in listdir:
    findCorners(os.path.join(_dir, _f))
    #matchArea(_src, os.path.join(_dst, _f))
