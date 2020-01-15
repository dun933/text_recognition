import os, sys, cv2
import glob
import numpy as np
import time
import random

def posibility_helper(p):
    def _wrapper(func):
        def _test(*args, **kwargs):
            cri = p*10
            rand = random.randint(0, 10)
            if cri >= rand:
                res = func(*args, **kwargs)
                return res
            return args[0]
        return _test
    return _wrapper


def resize(img, size):
    """
    size = (w, h)
    """
    h, w = img.shape[:2]
    if h < size[1] or w < size[0]:
        return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    else:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

@posibility_helper(p=0.7)
def rotate(img, max_angle, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101):
    height, width = img.shape[:2]
    angles = np.arange(-max_angle, max_angle, 0.5)
    angle = np.random.choice(angles)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (width, height), flags=interpolation, borderMode=border_mode)
    return img

@posibility_helper(p=0.4)
def zoom_in_or_out(img, scale=[-0.25, 0.2]):
    if isinstance(scale, float) or isinstance(scale, int):
        scale = [scale * -1, scale]
    h, w = img.shape[:2]
    range_ = np.arange(scale[0], scale[1], 0.01)
    scale = np.random.choice(range_)
    new_h = int(h * (1 + scale))
    new_w = int(w * (1 + scale))
    img_ = resize(img, size=(new_w, new_h))
    x1, y1 = int(abs(w-new_w)/2), int(abs(h-new_h)/2)
    x2, y2 = x1 + min(w, new_w), y1 + min(h, new_h)
    dst = img.copy()
    if scale < 0:
        dst[y1:y2, x1:x2]=img_
        mask = np.ones(dst.shape, np.uint8) * 255
        mask[y1:y2, x1:x2] = 0
        dst = cv2.inpaint(dst,mask,3,cv2.INPAINT_TELEA)
    else:
        dst = img_[y1:y2, x1:x2]
    return dst

@posibility_helper(p=0.2)
def low_resolution(img):
    img = cv2.pyrDown(img)
    return cv2.pyrUp(img)

@posibility_helper(p=0.2)
def blur(img, ksize=3):

    def _blur(img, ksize=ksize):
        return cv2.blur(img, (ksize, ksize))
    def _gaussian_blur(img, ksize=ksize):
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)
    def _median_blur(img, ksize=ksize):
        return cv2.medianBlur(img, ksize)
    def _motion_blur(img, kernel=ksize):
        return cv2.filter2D(img, -1, kernel / np.sum(kernel))
    blurs = [_blur, _gaussian_blur, _median_blur, _motion_blur]
    func = np.random.choice(blurs)
    return func(img)
   


@posibility_helper(p=0.4)
def optical_distortion(img, k=0, dx=0, dy=0, border_mode=cv2.BORDER_REFLECT_101):
    """
    k <- under 1
    dx <- move x coords +(move to left)
    dy <- move y coords +(move to top)
    """
    try:
        height, width = img.shape[:2]

        fx = width
        fy = width

        dx = np.random.choice(np.arange(-dx, dx))
        dy = np.random.choice(np.arange(-dy, dy))

        cx = width * 0.5 + dx
        cy = height * 0.5 + dy

        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float32)

        distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
        interpolation = np.random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_AREA])
        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height), cv2.CV_32FC1)
        img = cv2.remap(img, map1, map2, interpolation=interpolation, borderMode=border_mode)
    except Exception:
        pass
    return img


@posibility_helper(p=0.2)
def invert(img):
    return cv2.bitwise_not(img)

@posibility_helper(p=0.4)
def shift(img, max_range=0.25):
    from skimage import io
    from skimage import transform as tf

    image = img
    shear = np.random.choice(np.arange(-max_range, max_range, 0.05))
    afine_tf = tf.AffineTransform(shear=shear)
    modified = tf.warp(image, inverse_map=afine_tf)
    return modified


@posibility_helper(p=0.2)
def jpeg_compression(img, quality=50):
    q_val = np.random.choice(np.arange(quality, 90, 10))
    enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(q_val)]
    _, encoded_img = cv2.imencode('.jpg', img, enc_param)
    return cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

def total_augment(img):
    # img = rotate(img, max_angle=10)
    # img = zoom_in_or_out(img, scale=[-0.1, 0.2])
    if img.shape[1] >= 30:
        img = low_resolution(img)
    # img = blur(img, ksize=random.randint(3, 5))
    try:
        img = invert(img)
        img = jpeg_compression(img, quality=60)
        img = optical_distortion(img, dx=1, dy = 4)
    except Exception:
        pass
    # img = shift(img, max_range=0.2)
    return img

