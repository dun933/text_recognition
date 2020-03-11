import cv2
import numpy as np

def visualize_boxes(path_config_file, img, debug=False):
    with open(path_config_file, 'r+') as readf:
        count = 0
        for line in readf:
            count += 1
            list_inf = line.split()
            if len(list_inf) == 6:
                bb = [int(list_inf[2]), int(list_inf[3]), int(list_inf[2]) + int(list_inf[4]),
                      int(list_inf[3]) + int(list_inf[5])]
                wh_ratio = float(list_inf[4]) / float(list_inf[5])
                #print(count, list_inf[0], 'wh_ratio', wh_ratio)
                #cv2.putText(img, list_inf[0], (bb[0], bb[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2,
                #            cv2.LINE_AA)
                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)

    img_res = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
    if debug:
        cv2.imshow('result',img_res)
        cv2.waitKey(0)
    return img_res

def erose(img_list_path):
    for img_path in img_list_path:
        img = cv2.imread(img_path,0)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(img,kernel,iterations = 1)
        cv2.imshow('erode', erosion)
        ch = cv2.waitKey(0)
        if ch == 27:
            print('Saved',img_path)
            cv2.imwrite(img_path,erosion)

if __name__ == "__main__":
    img = cv2.imread('template_VIB/0001_ori.jpg')
    #visualize_boxes('template_VIB_page1.txt', img, debug=True)
    img_list=['background_VIB/IDCARD.jpg','background_VIB/OLD_IDCARD.jpg','background_VIB/DEPENDANT_PERSON.jpg']
    img_list=['background_VIB/RELATIVE.jpg']
    erose(img_list)


