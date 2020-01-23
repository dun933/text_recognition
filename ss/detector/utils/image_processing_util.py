import statistics
import time

import cv2, math
import numpy as np
import math
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import statistics
from .rotation import get_rotation_matrix, get_rotated_img

import time


def runtime(method):
    def runtime(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('[RUNTIME] ======> {%r}  %2.2f sec' % (method.__name__, te - ts))
        return result

    return runtime


def auto_contrast(img, clip=0):
    """ Written by WY. 18/09/11
    pixel의 intensity 분포를 파악하여(histogram) clip percentage 만큼의 부분을 좌우에서 도려내어
    adjust함.

    Parameters
    ----------
    img : (N, M[, ..., P]) ndarray
    clip : int
        histogram 좌/우를 얼마나 제거할 것인지

    Returns
    -------
    dst : (N, M[, ..., P]) ndarray
        contrast와 brightness가 조정된 이미지

    Examples
    --------
    >>> from image_processing_util from auto_contrast
    >>> image = cv2.imread("")
    >>> adujsted_image = auto_contrast(image)

    Reference
    --------
    http://answers.opencv.org/question/75510/how-to-make-auto-adjustmentsbrightness-and-contrast-for-image-android-opencv-image-correction/

    """

    gray = img.copy()

    hist_size = 256
    min_gray = 0
    max_gray = 0
    if clip == 0:
        min_gray, max_gray, _, _ = cv2.minMaxLoc(gray)
    else:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256], True, False)

        accumulator = [None] * hist_size
        accumulator[0] = hist[0][0]
        for i in range(1, hist_size):
            accumulator[i] = accumulator[i - 1] + hist[i][0]

        max_v = accumulator[-1]
        clip *= (max_v / 100.0)
        clip /= 2.0

        min_gray = 0
        while accumulator[min_gray] < clip:
            min_gray += 1

        max_gray = hist_size - 1
        while accumulator[max_gray] >= (max_v - clip):
            max_gray -= 1

    input_range = max_gray - min_gray
    if input_range < 50:
        return contrastStretching(img)
    alpha = (hist_size - 1) / input_range
    beta = -min_gray * alpha

    # min(O) = alpha * min(I) + beta
    dst = (gray * alpha) + beta
    dst = np.clip(dst, 0, 255)

    dst = dst.astype(np.uint8)
    return dst


## skew detetor : 이미지 rotation check
def skew_detect(input_img):
    """ Written by WY. 각도가 틀어지고 회전된 이미지를 원상태로 복원하여 반환.

    Parameters
    ----------
    input_img : (N, M[, ..., P]) ndarray

    Returns
    -------
    rotated_img : numpy.ndarray
        회전상태가 보정된 numpy.ndarray 형태의 이미지.
    skew_flag : boolean
        실제 input image의 보정이 이루어졌는지에 대한 boolean 타입의 flag

    Examples
    --------
    >>> from image_processing_util from skew_detect
    >>> image = cv2.imread("")
    >>> deskewed_img, skew_flag = skew_detect(image)
    """

    def __image_type_check(image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def get_max_freq_elem(arr):
        max_arr = []
        freqs = {}
        for i in arr:
            if i in freqs:
                freqs[i] += 1
            else:
                freqs[i] = 1

        sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
        max_freq = freqs[sorted_keys[0]]

        for k in sorted_keys:
            if freqs[k] == max_freq:
                max_arr.append(k)
        return max_arr

    def compare_sum(value):
        if value >= 44 and value <= 46:
            return True
        else:
            return False

    def calculate_deviation(angle):
        angle_in_degrees = np.abs(angle)
        deviation = np.abs((np.pi / 4) - angle_in_degrees)
        return deviation

    def determine_skew(input_img, sigma=3.0, num_peaks=20):  # sub method
        val, _ = cv2.threshold(input_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        high_thresh_val = val
        lower_thresh_val = val * 0.5
        edges = cv2.Canny(input_img, lower_thresh_val, high_thresh_val)
        edges[np.where(edges[:, :] == 255)] = 1
        edges = edges.astype('bool')

        h, a, d = hough_line(edges)
        _, ap, _ = hough_line_peaks(h, a, d, num_peaks=num_peaks)

        if len(ap) == 0:
            print("Bad Quality")
            return 0

        absolute_deviations = [calculate_deviation(k) for k in ap]
        average_deviation = np.mean(np.rad2deg(absolute_deviations))
        ap_deg = [np.rad2deg(x) for x in ap]

        bin_0_45 = []
        bin_45_90 = []
        bin_0_45n = []
        bin_45_90n = []

        for ang in ap_deg:
            deviation_sum = int(90 - ang + average_deviation)
            if compare_sum(deviation_sum):
                bin_45_90.append(ang)
                continue

            deviation_sum = int(ang + average_deviation)
            if compare_sum(deviation_sum):
                bin_0_45.append(ang)
                continue

            deviation_sum = int(-ang + average_deviation)
            if compare_sum(deviation_sum):
                bin_0_45n.append(ang)
                continue

            deviation_sum = int(90 + ang + average_deviation)
            if compare_sum(deviation_sum):
                bin_45_90n.append(ang)

        angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
        lmax = 0

        for j in range(len(angles)):
            l = len(angles[j])
            if l > lmax:
                lmax = l
                maxi = j

        if lmax:
            ans_arr = get_max_freq_elem(angles[maxi])
            ans_res = np.mean(ans_arr)

        else:
            ans_arr = get_max_freq_elem(ap_deg)
            ans_res = np.mean(ans_arr)

        rot_angle = ans_res

        if rot_angle >= 45: rot_angle -= 90
        if rot_angle <= -45: rot_angle += 90

        return rot_angle

    def Rotation(img, thetaRad):
        new_width = abs(math.sin(thetaRad)) * np.size(img, 0) + abs(math.cos(thetaRad)) * np.size(img, 1)
        new_height = abs(math.cos(thetaRad)) * np.size(img, 0) + abs(math.sin(thetaRad)) * np.size(img, 1)
        rot_mat = cv2.getRotationMatrix2D((new_width * .5, new_height * .5), thetaRad * 180 / math.pi, 1)

        pos = np.zeros((3, 1))
        # pos[0][0] = ((new_width - np.size(img, 1)) * .5)
        # pos[1][0] = ((new_height - np.size(img, 0)) * .5)
        pos.itemset(0, 0, ((new_width - np.size(img, 1)) * .5))
        pos.itemset(1, 0, ((new_height - np.size(img, 0)) * .5))

        res = np.dot(rot_mat, pos)

        # rot_mat[0][2] = res[0] + rot_mat[0][2]
        # rot_mat[1][2] = res[1] + rot_mat[1][2]
        rot_mat.itemset(0, 2, res[0] + rot_mat.item(0, 2))
        rot_mat.itemset(1, 2, res[1] + rot_mat.item(1, 2))
        rotated = cv2.warpAffine(img, rot_mat, (int(new_width), int(new_height)), cv2.INTER_LANCZOS4)

        del rot_mat
        del pos
        del res
        return rotated

    ## end : sub-method

    input_img_gray = __image_type_check(input_img)
    angle = determine_skew(cv2.pyrDown(input_img_gray.copy()))
    # angle = determine_skew(input_img_gray)

    if abs(angle) < 0.5: return input_img_gray, False
    rotated_img = Rotation(input_img, angle * math.pi / 180)
    return rotated_img, True

    def Rotation(img, thetaRad):
        new_width = abs(math.sin(thetaRad)) * np.size(img, 0) + abs(math.cos(thetaRad)) * np.size(img, 1)
        new_height = abs(math.cos(thetaRad)) * np.size(img, 0) + abs(math.sin(thetaRad)) * np.size(img, 1)
        rot_mat = cv2.getRotationMatrix2D((new_width * .5, new_height * .5), thetaRad * 180 / math.pi, 1)

        pos = np.zeros((3, 1))
        # pos[0][0] = ((new_width - np.size(img, 1)) * .5)
        # pos[1][0] = ((new_height - np.size(img, 0)) * .5)
        pos.itemset(0, 0, ((new_width - np.size(img, 1)) * .5))
        pos.itemset(1, 0, ((new_height - np.size(img, 0)) * .5))

        res = np.dot(rot_mat, pos)

        # rot_mat[0][2] = res[0] + rot_mat[0][2]
        # rot_mat[1][2] = res[1] + rot_mat[1][2]
        rot_mat.itemset(0, 2, res[0] + rot_mat.item(0, 2))
        rot_mat.itemset(1, 2, res[1] + rot_mat.item(1, 2))
        rotated = cv2.warpAffine(img, rot_mat, (int(new_width), int(new_height)), cv2.INTER_LANCZOS4)

        del rot_mat
        del pos
        del res
        return rotated

    ## end : sub-method

    input_img_gray = __image_type_check(input_img)

    angle = determine_skew(input_img_gray)
    print(angle)
    if abs(angle) < 0.5: return input_img_gray, False
    rotated_img = Rotation(input_img, angle * math.pi / 180)
    return rotated_img, True


## skew detetor : 이미지 rotation check
def skew_detect2(input_img):
    """ Written by WY. 각도가 틀어지고 회전된 이미지를 원상태로 복원하여 반환.

    Parameters
    ----------
    input_img : (N, M[, ..., P]) ndarray

    Returns
    -------
    rotated_img : numpy.ndarray
        회전상태가 보정된 numpy.ndarray 형태의 이미지.
    skew_flag : boolean
        실제 input image의 보정이 이루어졌는지에 대한 boolean 타입의 flag

    Examples
    --------
    >>> from image_processing_util from skew_detect
    >>> image = cv2.imread("")
    >>> deskewed_img, skew_flag = skew_detect(image)
    """

    def __image_type_check(image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def get_max_freq_elem(arr):
        max_arr = []
        freqs = {}
        for i in arr:
            if i in freqs:
                freqs[i] += 1
            else:
                freqs[i] = 1

        sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
        max_freq = freqs[sorted_keys[0]]

        for k in sorted_keys:
            if freqs[k] == max_freq:
                max_arr.append(k)
        return max_arr

    def compare_sum(value):
        if value >= 44 and value <= 46:
            return True
        else:
            return False

    def calculate_deviation(angle):
        angle_in_degrees = np.abs(angle)
        deviation = np.abs((np.pi / 4) - angle_in_degrees)
        return deviation

    def determine_skew(input_img, sigma=3.0, num_peaks=20):  # sub method
        val, _ = cv2.threshold(input_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        high_thresh_val = val
        lower_thresh_val = val * 0.5
        edges = cv2.Canny(input_img, lower_thresh_val, high_thresh_val)
        edges[np.where(edges[:, :] == 255)] = 1
        edges = edges.astype('bool')

        h, a, d = hough_line(edges)
        _, ap, _ = hough_line_peaks(h, a, d, num_peaks=num_peaks)

        if len(ap) == 0:
            print("Bad Quality")
            return 0

        absolute_deviations = [calculate_deviation(k) for k in ap]
        average_deviation = np.mean(np.rad2deg(absolute_deviations))
        ap_deg = [np.rad2deg(x) for x in ap]

        bin_0_45 = []
        bin_45_90 = []
        bin_0_45n = []
        bin_45_90n = []

        for ang in ap_deg:
            deviation_sum = int(90 - ang + average_deviation)
            if compare_sum(deviation_sum):
                bin_45_90.append(ang)
                continue

            deviation_sum = int(ang + average_deviation)
            if compare_sum(deviation_sum):
                bin_0_45.append(ang)
                continue

            deviation_sum = int(-ang + average_deviation)
            if compare_sum(deviation_sum):
                bin_0_45n.append(ang)
                continue

            deviation_sum = int(90 + ang + average_deviation)
            if compare_sum(deviation_sum):
                bin_45_90n.append(ang)

        angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
        lmax = 0

        for j in range(len(angles)):
            l = len(angles[j])
            if l > lmax:
                lmax = l
                maxi = j

        if lmax:
            ans_arr = get_max_freq_elem(angles[maxi])
            ans_res = np.mean(ans_arr)

        else:
            ans_arr = get_max_freq_elem(ap_deg)
            ans_res = np.mean(ans_arr)

        rot_angle = ans_res

        if rot_angle >= 45: rot_angle -= 90
        if rot_angle <= -45: rot_angle += 90

        return rot_angle

    def Rotation(img, thetaRad):
        new_width = abs(math.sin(thetaRad)) * np.size(img, 0) + abs(math.cos(thetaRad)) * np.size(img, 1)
        new_height = abs(math.cos(thetaRad)) * np.size(img, 0) + abs(math.sin(thetaRad)) * np.size(img, 1)
        rot_mat = cv2.getRotationMatrix2D((new_width * .5, new_height * .5), thetaRad * 180 / math.pi, 1)

        pos = np.zeros((3, 1))
        # pos[0][0] = ((new_width - np.size(img, 1)) * .5)
        # pos[1][0] = ((new_height - np.size(img, 0)) * .5)
        pos.itemset(0, 0, ((new_width - np.size(img, 1)) * .5))
        pos.itemset(1, 0, ((new_height - np.size(img, 0)) * .5))

        res = np.dot(rot_mat, pos)

        # rot_mat[0][2] = res[0] + rot_mat[0][2]
        # rot_mat[1][2] = res[1] + rot_mat[1][2]
        rot_mat.itemset(0, 2, res[0] + rot_mat.item(0, 2))
        rot_mat.itemset(1, 2, res[1] + rot_mat.item(1, 2))
        rotated = cv2.warpAffine(img, rot_mat, (int(new_width), int(new_height)), cv2.INTER_LANCZOS4)

        del rot_mat
        del pos
        del res
        return rotated
        ## end : sub-method

    input_img_gray = __image_type_check(input_img)
    angle = determine_skew(cv2.pyrDown(input_img_gray.copy()))

    angle_rad = angle * math.pi / 180

    if abs(angle) < 0.5:
        return input_img_gray, False, angle, None

    rotated_img, rot_matrix = get_rotated_img(input_img, angle_rad)

    return rotated_img, True, angle, rot_matrix


def Rotation_img(img, thetaRad):
    new_width = abs(math.sin(thetaRad)) * np.size(img, 0) + abs(math.cos(thetaRad)) * np.size(img, 1)
    new_height = abs(math.cos(thetaRad)) * np.size(img, 0) + abs(math.sin(thetaRad)) * np.size(img, 1)
    rot_mat = cv2.getRotationMatrix2D((new_width * .5, new_height * .5), thetaRad * 180 / math.pi, 1)

    pos = np.zeros((3, 1))
    pos.itemset(0, 0, ((new_width - np.size(img, 1)) * .5))
    pos.itemset(1, 0, ((new_height - np.size(img, 0)) * .5))

    res = np.dot(rot_mat, pos)
    rot_mat.itemset(0, 2, res[0] + rot_mat.item(0, 2))
    rot_mat.itemset(1, 2, res[1] + rot_mat.item(1, 2))

    rotated = cv2.warpAffine(img, rot_mat, (int(new_width), int(new_height)), cv2.INTER_LANCZOS4)

    del pos
    del res
    return rotated


def restore_point(cp1, cp2, pt, angle):
    x, y = pt
    p, q = cp1
    p_, q_ = cp2
    angle_ = -math.radians(angle)
    new_x_from_ori_img = int((x - p) * math.cos(angle_) - (y - q) * math.sin(angle_) + p)
    new_y_from_ori_img = int((x - p) * math.sin(angle_) + (y - q) * math.cos(angle_) + q)
    X = new_x_from_ori_img + (p_ - p)
    Y = new_y_from_ori_img + (q_ - q)
    return X, Y


def zoomScaleFinder(img, h=float(2000.0), AVG_RATIO=0.8, DEBUG=False):
    """ Written by WY. 이미지 내의 text를 segmentation하여 SSD가 detect 할 수 있는 최적의 글자 높이 배율을 찾아냄.

    Parameters
    ----------
    img : (N, M[, ..., P]) ndarray
    h   : float
        글자를 찾기 위한 고정 이미지 크기
    avg_ratio : float
    DEBUG : bool
        debugging용 이미지 저장 flag.

    Returns
    -------
    avg_character_height / ratio : numpy.ndarray
        해당 이미지 글자들의 평균 높이
    avg_character_height : int
        해당 이미지 글자들의 평균 높이

    Examples
    --------
    >>> from image_processing_util import zoomScaleFinder
    >>> image = cv2.imread("")
    >>> ratio, character_height = zoomScalefinder(image)
    """
    img_h = img.shape[0]
    ratio = h / img_h
    w = int(img.shape[1] * ratio)

    img_ = cv2.resize(img, (w, int(h)), interpolation=cv2.INTER_CUBIC)

    gray = img_
    if len(img_.shape) > 2:
        gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

    img_sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, img_threshold = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))

    img_morph = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, element)

    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(img_morph, 0, 1)
    else:
        contours, _ = cv2.findContours(img_morph, 0, 1)

    w_list = []

    for cc in contours:
        if cc.shape[0] < 100: continue
        approxCurve = cv2.approxPolyDP(cc, 3, True)
        bx, by, bw, bh = cv2.boundingRect(approxCurve)
        if bw > bh:
            if DEBUG: cv2.rectangle(img_, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            w_list.append(bh)

    w_list.sort()
    sum_v = 0
    cnt = 0
    for i in range(int(len(w_list) * (1 - AVG_RATIO)), int(len(w_list) * AVG_RATIO)):
        sum_v += w_list[i]
        cnt += 1

    if sum_v == 0 or len(w_list) == 0:
        print("whatever")

    avg_character_height = int(sum_v / cnt) if cnt > 0 else 0
    if DEBUG: cv2.imwrite("./debug/font_size_ret.png", img_)

    return avg_character_height / ratio, avg_character_height


def contrastStretching(img):
    """ Written by WY. 이미지 내의 Contrast 값을 0 - 255 범위 값으로 조정.

    Parameters
    ----------
    img : (N, M[, ..., P]) ndarray

    Returns
    -------
    img : (N, M[, ..., P]) ndarray
        contrast가 조정된 이미지

    Examples
    --------
    >>> from image_processing_util import contrastStretching
    >>> image = cv2.imread("")
    >>> contrast_image = contrastStrectching(image)
    """
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    max_v = np.max(img)
    min_v = np.min(img)

    img = (img - min_v) / (max_v - min_v) * 255
    img = img.astype(np.uint8)

    return img


def norm(img, k_size=9):
    math_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    math_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, math_kernel)
    math_div = img / math_close
    math_res = np.zeros(math_div.shape)
    cv2.normalize(math_div, math_res, 0, 255, cv2.NORM_MINMAX)
    return math_res.astype(np.uint8)


def ch_finder(img, h=float(2000.0), AVG_RATIO=0.8):
    ''' 이미지 글자 찾기 수정 버전. 최종 아님.
    '''
    h, w = img.shape
    flag = False
    if np.min(img.shape) > 3000:
        img = cv2.pyrDown(img)
        flag = True

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (33, 1))
    img_ = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = 255 - cv2.absdiff(img, img_)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 33))
    img_ = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = 255 - cv2.absdiff(img, img_)

    img = cv2.GaussianBlur(img, (3, 3), 1)
    img = norm(img, k_size=21)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 6))
    img_ = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)

    w_list = []
    _, contours, _ = cv2.findContours(255 - img_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_ = cv2.cvtColor(img_, cv2.COLOR_GRAY2RGB)
    for cc in contours:
        #         if cc.shape[0] < 100 : continue
        approxCurve = cv2.approxPolyDP(cc, 3, True)
        bx, by, bw, bh = cv2.boundingRect(approxCurve)
        if bw > bh:
            w_list.append(bh)
    #     return img_
    w_list.sort()
    sum_v = 0
    cnt = 0
    for i in range(int(len(w_list) * (1 - AVG_RATIO)), int(len(w_list) * AVG_RATIO)):
        sum_v += w_list[i]
        cnt += 1
    if cnt == 0:
        del img
        return 1

    avg_character_height = int(sum_v / cnt)
    median = statistics.median(w_list)
    if flag == True: median = median * 2
    return int(median)


def remove_high_freq(img, d=50):
    """ Written by WY. 해당 영역의 고주파를 제거. table의 음영이나 그림자 같은 noise를 제거하기 위해 사용.

    Parameters
    ----------
    img : (N, M[, ..., P]) ndarray
    d : int
        고주파를 제거할 영역

    Returns
    -------
    th : (N, M[, ..., P]) ndarray
        회전상태가 보정된 numpy.ndarray 형태의 이미지.

    Examples
    --------
    >>> from virtual_tables_recognizer from remove_high_freq
    >>> image = cv2.imread("")
    >>> non_high_freq_img = remove_high_freq(image, d=10)
    """
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    # 아래는 d 사이지의 사각형을 생성한 후, 사각형 바깥쪽을 제거하는 형태임.
    # 즉, 고주파영역을 제거하게 됨.
    # d값이 작을수록 사각형이 작고, 바깥영역 즉, 고주파영역이  많이 제거되기 때문에 이미지가 뭉게지고
    # d값이 클수록 사각형이 크고, 바깥영역 즉, 고주파 영역이 적게 제거되기 때문에 원래 이미지와 가까워짐.
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - d:crow + d, ccol - d:ccol + d] = 1
    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = img_back.astype(np.uint8)
    th = cv2.threshold(img_back, 125, 255, cv2.THRESH_OTSU)[1]

    return th