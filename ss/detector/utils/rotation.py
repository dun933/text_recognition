import cv2, numpy as np, math


def get_rotation_matrix(img_w, img_h, angle_rad):

    new_width = abs(math.sin(angle_rad)) * img_h + abs(math.cos(angle_rad)) * img_w
    new_height = abs(math.cos(angle_rad)) * img_h + abs(math.sin(angle_rad)) * img_w
    rot_mat = cv2.getRotationMatrix2D((new_width * .5, new_height * .5), angle_rad * 180 / math.pi, 1)

    pos = np.zeros((3, 1))
    pos.itemset(0, 0, ((new_width - img_w) * .5))
    pos.itemset(1, 0, ((new_height - img_h) * .5))

    res = np.dot(rot_mat, pos)

    rot_mat.itemset(0, 2, res[0] + rot_mat.item(0, 2))
    rot_mat.itemset(1, 2, res[1] + rot_mat.item(1, 2))

    return rot_mat

def deg_to_rad(angle):
    return angle * math.pi / 180

def get_rotated_img(img, angle):
    """


    :param img:
    :param angle: unit is rad
    :return:
    """

    img_h = img.shape[0]
    img_w = img.shape[1]


    rot_matrix = get_rotation_matrix(img_w, img_h, angle)

    out_w, out_h = get_output_img_size(img_w, img_h, angle)

    print("rot_matrix: {}".format(rot_matrix))

    output_img = cv2.warpAffine(img, rot_matrix, (int(out_w), int(out_h)), cv2.INTER_LANCZOS4)

    return output_img, rot_matrix






def get_output_img_size(img_w, img_h, angle):
    """


    :param img_w:
    :param img_h:
    :param angle: unit is rad
    :return:
    """

    new_width = abs(math.sin(angle)) * img_h + abs(math.cos(angle)) * img_w
    new_height = abs(math.cos(angle)) * img_h + abs(math.sin(angle)) * img_w

    return new_width, new_height