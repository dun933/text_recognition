# Script to convert yolo annotations to voc format

# Sample format
# <annotation>
#     <folder>_image_fashion</folder>
#     <filename>brooke-cagle-39574.jpg</filename>
#     <size>
#         <width>1200</width>
#         <height>800</height>
#         <depth>3</depth>
#     </size>
#     <segmented>0</segmented>
#     <object>
#         <name>head</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>549</xmin>
#             <ymin>251</ymin>
#             <xmax>625</xmax>
#             <ymax>335</ymax>
#         </bndbox>
#     </object>
# <annotation>
import os
import xml.etree.cElementTree as ET
from PIL import Image
import codecs

ANNOTATIONS_DIR_PREFIX = "annotations"

DESTINATION_DIR = "converted_labels"

CLASS_MAPPING = {
    '0': 'À',
    '1': 'Á',
    '2': 'Â',
    '3': 'Ã',
    '4': 'È',
    '5': 'É',
    '6': 'Ê',
    '7': 'Ì',
    '8': 'Í',
    '9': 'Ò',
    '10': 'Ó',
    '11': 'Ô',
    '12': 'Õ',
    '13': 'Ù',
    '14': 'Ú',
    '15': 'Ý',
    '16': 'à',
    '17': 'á',
    '18': 'â',
    '19': 'ã',
    '20': 'è',
    '21': 'é',
    '22': 'ê',
    '23': 'ì',
    '24': 'í',
    '25': 'ò',
    '26': 'ó',
    '27': 'ô',
    '28': 'õ',
    '29': 'ù',
    '30': 'ú',
    '31': 'ý',
    '32': 'Ă',
    '33': 'ă',
    '34': 'Đ',
    '35': 'đ',
    '36': 'Ĩ',
    '37': 'ĩ',
    '38': 'Ũ',
    '39': 'ũ',
    '40': 'Ơ',
    '41': 'ơ',
    '42': 'Ư',
    '43': 'ư',
    '44': 'Ạ',
    '45': 'ạ',
    '46': 'Ả',
    '47': 'ả',
    '48': 'Ấ',
    '49': 'ấ',
    '50': 'Ầ',
    '51': 'ầ',
    '52': 'Ẩ',
    '53': 'ẩ',
    '54': 'Ẫ',
    '55': 'ẫ',
    '56': 'Ậ',
    '57': 'ậ',
    '58': 'Ắ',
    '59': 'ắ',
    '60': 'Ằ',
    '61': 'ằ',
    '62': 'Ẳ',
    '63': 'ẳ',
    '64': 'Ẵ',
    '65': 'ẵ',
    '66': 'Ặ',
    '67': 'ặ',
    '68': 'Ẹ',
    '69': 'ẹ',
    '70': 'Ẻ',
    '71': 'ẻ',
    '72': 'Ẽ',
    '73': 'ẽ',
    '74': 'Ế',
    '75': 'ế',
    '76': 'Ề',
    '77': 'ề',
    '78': 'Ể',
    '79': 'ể',
    '80': 'Ễ',
    '81': 'ễ',
    '82': 'Ệ',
    '83': 'ệ',
    '84': 'Ỉ',
    '85': 'ỉ',
    '86': 'Ị',
    '87': 'ị',
    '88': 'Ọ',
    '89': 'ọ',
    '90': 'Ỏ',
    '91': 'ỏ',
    '92': 'Ố',
    '93': 'ố',
    '94': 'Ồ',
    '95': 'ồ',
    '96': 'Ổ',
    '97': 'ổ',
    '98': 'Ỗ',
    '99': 'ỗ',
    '100': 'Ộ',
    '101': 'ộ',
    '102': 'Ớ',
    '103': 'ớ',
    '104': 'Ờ',
    '105': 'ờ',
    '106': 'Ở',
    '107': 'ở',
    '108': 'Ỡ',
    '109': 'ỡ',
    '110': 'Ợ',
    '111': 'ợ',
    '112': 'Ụ',
    '113': 'ụ',
    '114': 'Ủ',
    '115': 'ủ',
    '116': 'Ứ',
    '117': 'ứ',
    '118': 'Ừ',
    '119': 'ừ',
    '120': 'Ử',
    '121': 'ử',
    '122': 'Ữ',
    '123': 'ữ',
    '124': 'Ự',
    '125': 'ự',
    '126': 'Ỳ',
    '127': 'ỳ',
    '128': 'Ỵ',
    '129': 'ỵ',
    '130': 'Ỷ',
    '131': 'ỷ',
    '132': 'Ỹ',
    '133': 'ỹ',
    '135': '0',
    '136': '1',
    '137': '2',
    '138': '3',
    '139': '4',
    '140': '5',
    '141': '6',
    '142': '7',
    '143': '8',
    '144': '9',
    '145': 'A',
    '146': 'B',
    '147': 'C',
    '148': 'D',
    '149': 'E',
    '150': 'F',
    '151': 'G',
    '152': 'H',
    '153': 'I',
    '154': 'J',
    '155': 'K',
    '156': 'L',
    '157': 'M',
    '158': 'N',
    '159': 'O',
    '160': 'P',
    '161': 'Q',
    '162': 'R',
    '163': 'S',
    '164': 'T',
    '165': 'U',
    '166': 'V',
    '167': 'W',
    '168': 'X',
    '169': 'Y',
    '170': 'Z',
    '171': 'a',
    '172': 'b',
    '173': ' ',
    '174': 'c',
    '175': 'd',
    '176': 'e',
    '177': 'f',
    '178': 'g',
    '179': 'h',
    '180': 'i',
    '181': 'j',
    '182': 'k',
    '183': 'l',
    '184': 'm',
    '185': 'n',
    '186': 'o',
    '187': 'p',
    '188': 'q',
    '189': 'r',
    '190': 's',
    '191': 't',
    '192': 'u',
    '193': 'v',
    '194': 'w',
    '195': 'x',
    '196': 'y',
    '197': 'z',
    '198': '',
    '199': '0',
    '200': '1',
    '201': '2',
    '202': '3',
    '203': '4',
    '204': '5',
    '205': '6',
    '206': '7',
    '207': '8',
    '208': '9',
    '209': ' ',
    '210': ' ',
    '211': '^',
    '212': '&',
    '213': "'",
    '214': '*',
    '215': ':',
    '216': ',',
    '217': '@',
    '218': '$',
    '219': '.',
    '220': '=',
    '221': '!',
    '222': '>',
    '223': '-',
    '224': '{',
    '225': '(',
    '226': '[',
    '227': '<',
    '228': '_',
    '229': '#',
    '230': '%',
    '231': '+',
    '232': '?',
    '233': '"',
    '234': '※',
    '235': '}',
    '236': ')',
    '237': ']',
    '238': ';',
    '239': '/',
    '240': '~',
    '241': '\\'
    # Add your remaining classes here.
}

import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

XML_EXT = '.xml'
ENCODE_METHOD = 'utf8'

class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize,databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            # name.text = ustr(each_object['name'])
            name.text = each_object['name']
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(float(each_object['ymax'])) == int(float(self.imgSize[0])) or (int(float(each_object['ymin']))== 1):
                truncated.text = "1" # max == height or min
            elif (int(float(each_object['xmax']))==int(float(self.imgSize[1]))) or (int(float(each_object['xmin']))== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str( bool(each_object['difficult']) & 1 )
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()

def save_xml(img_path, xml_path, object_list, img_w, img_h):
    path = os.path.normpath(img_path)
    path_splt = path.split(os.sep)
    file_prefix = path_splt[-1]
    # img = Image.open(img_path)
    # w, h = img.size
    writer = PascalVocWriter('images', file_prefix, (img_w, img_h, 3), localImgPath=img_path)
    difficult = 1
    for orddict in object_list:
        # print(orddict['coords'])
        _x = int(orddict['x1'] * img_w)
        _y = int(orddict['y1'] * img_h)
        _x2 = int(orddict['x2'] * img_w)
        _y2 = int(orddict['y2'] * img_h)

        class_type = orddict['class_type']
        char = orddict['char']
        writer.addBndBox(_x, _y, _x2, _y2, char, difficult)
    writer.save(xml_path)


def save_txt(img_path, object_list, zoom_rate=1):
    path = os.path.normpath(img_path)
    path_splt = path.split(os.sep)
    file_prefix = path_splt[-1][:-4]
    img = Image.open(img_path)
    w, h = img.size
    # writer = PascalVocWriter('Imgs', file_prefix, (w, h, 3), localImgPath=img_path)
    # difficult = 1
    list_line = []
    for obj in object_list:
        xmin = int(round(obj[0]/zoom_rate, 2))
        ymin = int(round(obj[1]/zoom_rate, 2))
        xmax = int(round(obj[2]/zoom_rate, 2))
        ymax = int(round(obj[3]/zoom_rate, 2))

        w = xmax - xmin
        h = ymax - ymin
        pred = round(obj[5], 2)

        if obj[4] != '':
            class_name = obj[4]
            # writer.addBndBox(xmin, ymin, xmax, ymax, obj[4], difficult)
        else:
            class_name = " "
            # obj[4] = ' '
            # writer.addBndBox(xmin, ymin, xmax, ymax, ' ', difficult)
        line = class_name + ' ' + str(pred) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(w) + ' ' + str(h)
        list_line.append(line)
    with codecs.open('/data/tuan-anh/aicr-TODAH/test_images/vn_test/Anno/{}.txt'.format(file_prefix), 'w', encoding='utf8') as filehandle:
        for line in list_line:
            filehandle.write('%s\n' % line)
    # writer.save('/data/tuan-anh/aicr-TODAH/test_images/vn_test/Anno/{}.txt'.format(file_prefix))


