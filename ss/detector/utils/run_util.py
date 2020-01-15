import cv2
import numpy as np

from matplotlib import rc, pyplot as plt
import matplotlib.patches as patches

try:
    import cairocffi as cairo
except:
    pass
import matplotlib

matplotlib.use('agg')


def drawAnnotation(coord_list, img, show_shape=False, show_conf=False, save_file_name=None, inch=50):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(inch, inch)
    plt.imshow(img, cmap='Greys_r')

    for c in coord_list:
        # print(str(c))
        # ax = fig.add_subplot(111, aspect='equal')
        class_nm = c.class_nm
        conf = c.confidence
        c = c.getAbsolute_coord()
        color = 'b'

        ax.add_patch(
            patches.Rectangle(
                (c[0], c[1]),
                c[2] - c[0],
                c[3] - c[1]
                , linewidth=1, edgecolor=color, facecolor='none'
                # , fill=False      # remove background
            )
        )
        if show_conf:
            plt.text(c[0], c[1], round(conf, 2), fontsize=max((c[3] - c[1]) / 6, 8), fontdict={"color": color})
    plt.show()

    if show_shape:
        print(img.shape)
    if save_file_name is not None:
        fig.savefig(save_file_name, bbox_inches='tight')


def calculateIOU_voting(point_obj_list, iou_threshold):
    applied_iou_list = []
    overlap_point_list = []
    candidate_dict = dict()
    rtn_list = []
    idx = len(point_obj_list)

    for i in range(idx):
        candidate_dict[i] = [i]

    for i in range(idx):
        point_obj = point_obj_list[i]

        for j in range(i + 1, idx):
            tmp_obj = point_obj_list[j]

            if point_obj.absoulte_coord[3] < tmp_obj.absoulte_coord[1]:
                break
            xA = max(point_obj.absoulte_coord[0], tmp_obj.absoulte_coord[0])
            yA = max(point_obj.absoulte_coord[1], tmp_obj.absoulte_coord[1])
            xB = min(point_obj.absoulte_coord[2], tmp_obj.absoulte_coord[2])
            yB = min(point_obj.absoulte_coord[3], tmp_obj.absoulte_coord[3])

            interArea = (xB - xA + 1) * (yB - yA + 1)
            if (xB - xA + 1) < 0 or (yB - yA + 1) < 0:
                # if interArea < 0 :
                interArea = 0

            boxAArea = (point_obj.absoulte_coord[2] - point_obj.absoulte_coord[0] + 1) \
                       * (point_obj.absoulte_coord[3] - point_obj.absoulte_coord[1] + 1)
            boxBArea = (tmp_obj.absoulte_coord[2] - tmp_obj.absoulte_coord[0] + 1) \
                       * (tmp_obj.absoulte_coord[3] - tmp_obj.absoulte_coord[1] + 1)

            iou = interArea / (boxAArea + boxBArea - interArea + 0.001)

            if iou > iou_threshold or interArea == boxBArea:
                # if iou > iou_threshold:

                id_ = (point_obj.confidence * 1.01 if point_obj.class_nm == "number" else point_obj.confidence) > \
                      (tmp_obj.confidence * 1.01 if tmp_obj.class_nm == "number" else tmp_obj.confidence) and j or i

                candidate = i if id_ == j else j

                candidate_dict[candidate].append(id_)

                overlap_point_list.append(id_)

    # print('before removing duplicated id cnt : ', str(len(overlap_point_list)))
    # redundant number removal
    overlap_point_list = list(set(overlap_point_list))
    # print('after removing duplicated id cnt : ', str(len(overlap_point_list)))

    overlap_point_list.sort()
    # print('overlap_point_list : ', str(overlap_point_list))

    for i in range(len(point_obj_list)):
        if i in overlap_point_list:
            continue
        applied_iou_list.append(i)

    def vote_class(boxes):
        classes = [point_obj_list[box].class_nm for box in boxes]
        counter = Counter(classes)
        top2 = counter.most_common(2)

        if len(top2) == 1:
            return top2[0][0]

        if top2[0][1] != top2[1][1]:
            return top2[0][0]

        else:
            max_zoom_ratio = max([point_obj_list[box].zoom_ratio for box in boxes])
            classes_top_zoom_ratio = [point_obj_list[box].class_nm for box in boxes if
                                      point_obj_list[box].zoom_ratio == max_zoom_ratio]
            counter_top_zoom_ratio = Counter(classes_top_zoom_ratio)
            top = counter_top_zoom_ratio.most_common(1)

            return top[0][0]

    for i in applied_iou_list:
        point = point_obj_list[i]
        point_class = vote_class(candidate_dict[i])
        point.class_nm = point_class
        rtn_list.append(point)

    return rtn_list


def calculateIOU(point_obj_list, iou_threshold):
    applied_iou_list = []
    overlap_point_list = []

    idx = len(point_obj_list)

    for i in range(idx - 1):
        point_obj = point_obj_list[i]

        for j in range(i + 1, idx):
            tmp_obj = point_obj_list[j]

            if point_obj.absoulte_coord[3] < tmp_obj.absoulte_coord[1]:
                break
            xA = max(point_obj.absoulte_coord[0], tmp_obj.absoulte_coord[0])
            yA = max(point_obj.absoulte_coord[1], tmp_obj.absoulte_coord[1])
            xB = min(point_obj.absoulte_coord[2], tmp_obj.absoulte_coord[2])
            yB = min(point_obj.absoulte_coord[3], tmp_obj.absoulte_coord[3])

            interArea = (xB - xA + 1) * (yB - yA + 1)
            if (xB - xA + 1) < 0 or (yB - yA + 1) < 0:
                # if interArea < 0 :
                interArea = 0

            boxAArea = (point_obj.absoulte_coord[2] - point_obj.absoulte_coord[0] + 1) \
                       * (point_obj.absoulte_coord[3] - point_obj.absoulte_coord[1] + 1)
            boxBArea = (tmp_obj.absoulte_coord[2] - tmp_obj.absoulte_coord[0] + 1) \
                       * (tmp_obj.absoulte_coord[3] - tmp_obj.absoulte_coord[1] + 1)

            iou = interArea / (boxAArea + boxBArea - interArea + 0.001)

            if iou > iou_threshold or interArea == boxBArea:
                # if iou > iou_threshold:

                id_ = (point_obj.confidence * 1.01 if point_obj.class_nm == "number" else point_obj.confidence) > \
                      (tmp_obj.confidence * 1.01 if tmp_obj.class_nm == "number" else tmp_obj.confidence) and j or i
                overlap_point_list.append(id_)

    # print('before removing duplicated id cnt : ', str(len(overlap_point_list)))
    # redundant number removal
    overlap_point_list = list(set(overlap_point_list))
    # print('after removing duplicated id cnt : ', str(len(overlap_point_list)))

    overlap_point_list.sort()
    # print('overlap_point_list : ', str(overlap_point_list))

    for i in range(len(point_obj_list)):
        point_obj = point_obj_list[i]
        if i in overlap_point_list:
            continue
        applied_iou_list.append(point_obj)

    return applied_iou_list


def calculateIOUCNX(point_obj_list, iou_threshold, confidence_threshold = 0.5, interarea_threshold = 0.4, bAoB_threshold = 0.9):
    applied_iou_list = []
    overlap_point_list = []

    idx = len(point_obj_list)
    overlap_p = [0]*idx

    for i in range(idx - 1):
        point_obj = point_obj_list[i]
        id_p = i
        if overlap_p[i] == 1:
            continue
        for j in range(i + 1, idx):
            if overlap_p[j] == 1:
                continue
            tmp_obj = point_obj_list[j]
            id_t = j

            if point_obj.absoulte_coord[3] < tmp_obj.absoulte_coord[1]:
                break
            if point_obj.absoulte_coord[2] < tmp_obj.absoulte_coord[0] or point_obj.absoulte_coord[0] > \
                    tmp_obj.absoulte_coord[2]:
                continue
            xA = max(point_obj.absoulte_coord[0], tmp_obj.absoulte_coord[0])
            yA = max(point_obj.absoulte_coord[1], tmp_obj.absoulte_coord[1])
            xB = min(point_obj.absoulte_coord[2], tmp_obj.absoulte_coord[2])
            yB = min(point_obj.absoulte_coord[3], tmp_obj.absoulte_coord[3])

            interArea = 0
            if (xB - xA + 1) > 0 and (yB - yA + 1) > 0:
                # if interArea < 0 :
                interArea = (xB - xA + 1) * (yB - yA + 1)

            boxAArea = (point_obj.absoulte_coord[2] - point_obj.absoulte_coord[0] + 1) \
                       * (point_obj.absoulte_coord[3] - point_obj.absoulte_coord[1] + 1)
            heightA = (point_obj.absoulte_coord[3] - point_obj.absoulte_coord[1] + 1)
            boxBArea = (tmp_obj.absoulte_coord[2] - tmp_obj.absoulte_coord[0] + 1) \
                       * (tmp_obj.absoulte_coord[3] - tmp_obj.absoulte_coord[1] + 1)
            heightB = (tmp_obj.absoulte_coord[3] - tmp_obj.absoulte_coord[1] + 1)

            iou = interArea / (boxAArea + boxBArea - interArea + 0.001)
            boxAoB =  boxAArea/ boxBArea

            if iou > iou_threshold or interArea / boxBArea > interarea_threshold or interArea / boxAArea > interarea_threshold:
                # if iou > iou_threshold:
                if abs(point_obj.confidence - tmp_obj.confidence) < confidence_threshold and iou < 0.8:
                    if heightA / heightB >= 0.8 and boxAoB > bAoB_threshold :
                        id_ = id_t
                    else:
                        id_ = id_p
                else:
                    id_ = (point_obj.confidence * 1.01 if point_obj.class_nm == "number" else point_obj.confidence) > \
                          (tmp_obj.confidence * 1.01 if tmp_obj.class_nm == "number" else tmp_obj.confidence) and id_t or id_p
                if id_ == id_p:
                    point_obj = point_obj_list[id_t]
                    id_p = id_t
                overlap_point_list.append(id_)
                overlap_p[id_] = 1

    # print('before removing duplicated id cnt : ', str(len(overlap_point_list)))
    # redundant number removal
    overlap_point_list = list(set(overlap_point_list))
    # print('after removing duplicated id cnt : ', str(len(overlap_point_list)))

    overlap_point_list.sort()
    # print('overlap_point_list : ', str(overlap_point_list))

    for i in range(len(point_obj_list)):
        point_obj = point_obj_list[i]
        if i in overlap_point_list:
            continue
        applied_iou_list.append(point_obj)

    return applied_iou_list


def calculateIOU2(point_obj_list, iou_threshold):
    """
    calculateIOU(point_obj_list, iou_threshold):
    """
    print("test")
    applied_iou_list = []
    overlap_point_list = []

    idx = len(point_obj_list)

    point_dict = {"hangul": [], "alphabet": [], "symbol": [], "number": []}

    for i in range(idx):
        point_obj = point_obj_list[i]
        if point_obj.class_nm == "hangul":
            point_dict["hangul"].append(point_obj)
        elif point_obj.class_nm == "alphabet":
            point_dict["alphabet"].append(point_obj)
        elif point_obj.class_nm == "symbol":
            point_dict["symbol"].append(point_obj)
        elif point_obj.class_nm == "number":
            point_dict["number"].append(point_obj)

    for k, v in point_dict.items():

        idx2 = len(v)
        overlap_point_list = []
        for i in range(idx2 - 1):
            point_obj = point_dict[k][i]

            for j in range(i + 1, idx2):
                tmp_obj = point_dict[k][j]

                if (point_obj.absoulte_coord[3] < tmp_obj.absoulte_coord[1]):
                    break
                xA = max(point_obj.absoulte_coord[0], tmp_obj.absoulte_coord[0])
                yA = max(point_obj.absoulte_coord[1], tmp_obj.absoulte_coord[1])
                xB = min(point_obj.absoulte_coord[2], tmp_obj.absoulte_coord[2])
                yB = min(point_obj.absoulte_coord[3], tmp_obj.absoulte_coord[3])

                width = max(xB - xA + 1, 0)
                height = max(yB - yA + 1, 0)
                interArea = width * height
                # interArea = (xB - xA + 1) * (yB - yA + 1)

                # if(xB - xA + 1) < 0 or (yB - yA + 1) < 0:
                # if interArea < 0 :
                #    interArea = 0

                boxAArea = (point_obj.absoulte_coord[2] - point_obj.absoulte_coord[0] + 1) \
                           * (point_obj.absoulte_coord[3] - point_obj.absoulte_coord[1] + 1)
                boxBArea = (tmp_obj.absoulte_coord[2] - tmp_obj.absoulte_coord[0] + 1) \
                           * (tmp_obj.absoulte_coord[3] - tmp_obj.absoulte_coord[1] + 1)

                iou = interArea / (boxAArea + boxBArea - interArea + 0.001)

                if iou > iou_threshold or interArea == boxBArea:
                    # if iou > iou_threshold:
                    id_ = point_obj.confidence > tmp_obj.confidence and j or i
                    overlap_point_list.append(id_)

        # print('before removing duplicated id cnt : ', str(len(overlap_point_list)))
        # redundant number removal
        overlap_point_list = list(set(overlap_point_list))
        # print('after removing duplicated id cnt : ', str(len(overlap_point_list)))

        overlap_point_list.sort()
        # print('overlap_point_list : ', str(overlap_point_list))

        for i, _ in enumerate(v):
            point_obj = v[i]
            if i in overlap_point_list:
                continue
            applied_iou_list.append(point_obj)

    return applied_iou_list


# context 내 특정 위치로 이동하여 지정된 폰트 타입, 폰트 크기, 기울임, 두께로 Text를 그림
def setTextToContext(context, x, y, text, size=30):
    result = []
    context.set_source_rgb(0, 0, 0)
    context.set_font_size(size)
    context.select_font_face('맑은 고딕', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    context.move_to(x, y)

    context.show_text(text)


def showLineUpDocument2(sceneTextList, ori_img, flag=0, save_file_name=None):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, ori_img.shape[1], ori_img.shape[0])
    context = cairo.Context(surface)
    context.set_source_rgb(1, 1, 1)  # White
    context.paint()
    mx = context.get_matrix()

    for text in sceneTextList:
        coord = text.aabb()
        if flag == 0:
            label = text.to_string()
        elif flag == 1:
            label = text.get_word()
        setTextToContext(context, coord[0], coord[1] + 30, label)

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    a.shape = (ori_img.shape[0], ori_img.shape[1], 4)
    a = a[:, :, 0]  # grab single channel
    a = a.astype(np.float32) / 255
    a = np.expand_dims(a, 0)

    fig, ax = plt.subplots(1)
    fig.set_size_inches(50, 50)

    plt.imshow(a[0], cmap='Greys_r')

    for c in sceneTextList:
        # print(str(c))
        # ax = fig.add_subplot(111, aspect='equal')
        c = c.aabb()
        ax.add_patch(
            patches.Rectangle(
                (c[0], c[1]),
                c[2] - c[0],
                c[3] - c[1]
                , linewidth=1, edgecolor='r', facecolor='none'
                # , fill=False      # remove background
            )
        )
    plt.show()
    if save_file_name is not None:
        fig.savefig(save_file_name)


def showLineUpDocument(sceneTextList, ori_img, flag=0, save_file_name=None):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, ori_img.shape[1], ori_img.shape[0])
    context = cairo.Context(surface)
    context.set_source_rgb(1, 1, 1)  # White
    context.paint()
    mx = context.get_matrix()

    for text in sceneTextList:
        coord = text.aabb()
        if flag == 0:
            label = text.to_string()
        elif flag == 1:
            label = text.get_word()
        setTextToContext(context, coord[0], coord[1] + 30, label)

    if save_file_name is not None:
        surface.write_to_png(save_file_name)
    else:
        buf = surface.get_data()
        a = np.frombuffer(buf, np.uint8)
        a.shape = (ori_img.shape[0], ori_img.shape[1], 4)
        a = a[:, :, 0]  # grab single channel
        a = a.astype(np.float32) / 255
        a = np.expand_dims(a, 0)

        fig, ax = plt.subplots(1)
        fig.set_size_inches(50, 50)

        plt.imshow(a[0], cmap='Greys_r')

        for c in sceneTextList:
            # print(str(c))
            # ax = fig.add_subplot(111, aspect='equal')
            c = c.aabb()
            ax.add_patch(
                patches.Rectangle(
                    (c[0], c[1]),
                    c[2] - c[0],
                    c[3] - c[1]
                    , linewidth=1, edgecolor='r', facecolor='none'
                    # , fill=False      # remove background
                )
            )

        plt.show()


def showTableDocument2(textList, ori_img, save_file_name=None, show_cellid=False, fontsize=30):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, ori_img.shape[1], ori_img.shape[0])
    context = cairo.Context(surface)
    context.set_source_rgb(1, 1, 1)  # White
    context.paint()
    mx = context.get_matrix()

    for t, text in enumerate(textList):
        if show_cellid:
            setTextToContext(context, text[0], text[1] + 30, '%d: %s' % (t, text[5]), size=fontsize)
        else:
            setTextToContext(context, text[0], text[1] + 30, '%s' % text[5], size=fontsize)

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    a.shape = (ori_img.shape[0], ori_img.shape[1], 4)
    a = a[:, :, 0]  # grab single channel
    a = a.astype(np.float32) / 255
    a = np.expand_dims(a, 0)

    fig, ax = plt.subplots(1)
    fig.set_size_inches(50, 50)

    plt.imshow(a[0], cmap='Greys_r')

    for c in textList:
        # print(str(c))
        # ax = fig.add_subplot(111, aspect='equal')
        ax.add_patch(
            patches.Rectangle(
                (c[0], c[1]),
                c[2] - c[0],
                c[3] - c[1]
                , linewidth=1, edgecolor='r', facecolor='none'
                # , fill=False      # remove background
            )
        )

    plt.show()
    if save_file_name is not None:
        fig.savefig(save_file_name)


def showTableDocument(textList, ori_img, save_file_name=None, show_cellid=False, fontsize=30):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, ori_img.shape[1], ori_img.shape[0])
    context = cairo.Context(surface)
    context.set_source_rgb(1, 1, 1)  # White
    context.paint()
    mx = context.get_matrix()

    for t, text in enumerate(textList):
        if show_cellid:
            setTextToContext(context, text[0], text[1] + 30, '%d: %s' % (t, text[5]), size=fontsize)
        else:
            setTextToContext(context, text[0], text[1] + 30, '%s' % text[5], size=fontsize)

    if save_file_name is not None:
        surface.write_to_png(save_file_name)
    else:
        buf = surface.get_data()
        a = np.frombuffer(buf, np.uint8)
        a.shape = (ori_img.shape[0], ori_img.shape[1], 4)
        a = a[:, :, 0]  # grab single channel
        a = a.astype(np.float32) / 255
        a = np.expand_dims(a, 0)

        fig, ax = plt.subplots(1)
        fig.set_size_inches(50, 50)

        plt.imshow(a[0], cmap='Greys_r')

        for c in textList:
            # print(str(c))
            # ax = fig.add_subplot(111, aspect='equal')
            ax.add_patch(
                patches.Rectangle(
                    (c[0], c[1]),
                    c[2] - c[0],
                    c[3] - c[1]
                    , linewidth=1, edgecolor='r', facecolor='none'
                    # , fill=False      # remove background
                )
            )

        plt.show()


def read_file(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    return data


def write_file(data, filename):
    with open(filename, 'wb') as f:
        f.write(data)


def character_check(list_, bx, by, bx2, by2):
    chs = [ch for ch in list_ if (ch[0] > bx) & (ch[1] > by) & (ch[2] < bx2) & (ch[3] < by2)]
    upper = 0;
    lower = 0
    for f in chs:
        if f[4].isupper():
            upper += 1
        elif f[4].islower():
            lower += 1
    return upper, lower


# cuongnd
def draw_annot_by_text(img_path, txt_file, save_img_path, isGT=True, inch=50):
    with open(txt_file) as f:
        content = f.readlines()

    fig, ax = plt.subplots(1)
    fig.set_size_inches(inch, inch)
    ori_img = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)

    plt.imshow(img, cmap='Greys_r')
    for c in content:
        print(str(c))
        obj = c.split(" ")
        class_nm = obj[0]
        if class_nm == 'vietnam':
            color = 'g'
        elif class_nm == 'alphabet':
            color = 'b'
        elif class_nm == 'number':
            color = 'r'
        else:
            color = 'c'  # symbol
        if isGT:
            conf = obj[1]
            xmin = obj[2]
            ymin = obj[3]
            width = obj[4] - obj[2]
            height = obj[5] - obj[3]
        else:
            xmin = obj[1]
            ymin = obj[2]
            width = obj[3] - obj[1]
            height = obj[4] - obj[2]
            plt.text(xmin, ymin, round(conf, 2), fontsize=max(width / 6, 8), fontdict={"color": color})

        ax.add_patch(patches.Rectangle((xmin, ymin), width, height,
                                       linewidth=1, edgecolor=color, facecolor='none'))

    plt.show()

    if save_img_path is not None:
        fig.savefig(save_img_path)
