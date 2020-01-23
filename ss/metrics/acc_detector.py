import argparse
import glob
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
import shutil
# from argparse import RawTextHelpFormatter
import sys
# reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('UTF8')
import _init_paths
from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import BBFormat
import codecs

GT_dir = '/home/advlab/data/test_vn/test_image_vn/v5/ground_truth'
# GT_dir = '/data/CuongND/aicr_dssd_train/outputs/predict_2019-10-11_11-32_9720_ignore_signature_barcode'
# DT_dir = '/home/advlab/data/test_vn/test_image_vn/v5/ground_truth'
DT_dir = '/data/CuongND/aicr_dssd_train/outputs/predict_2019-10-11_11-29_9674_raw'
# DT_dir = '/data/CuongND/aicr_dssd_train/outputs/predict_2019-10-11_11-32_9720_ignore_signature_barcode'


file_list = [
    '20190731_144554',
    '20190731_144540',
    '190715070245517_8478000669_pod',
    '190715070249216_8477872491_pod',
    '190715070317353_8479413342_pod'
]


# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(
            'argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' % argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append('argument %s: required argument' % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append(
                '%s. It must be in the format \'width,height\' (e.g. \'600,400\')' % errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg


def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    files = [file for file in files if file.replace(".txt", "") in file_list]
    # print(files)
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = codecs.open(f, "r", encoding='utf-8-sig')
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                if splitLine[0] != '':
                    idClass = (splitLine[0])
                    x = float(splitLine[1])
                    y = float(splitLine[2])
                    w = float(splitLine[3])
                    h = float(splitLine[4])
                    # x = float(splitLine[2])
                    # y = float(splitLine[3])
                    # w = float(splitLine[4])
                    # h = float(splitLine[5])
                    bb = BoundingBox(
                        nameOfImage,
                        idClass,
                        x,
                        y,
                        w,
                        h,
                        coordType,
                        imgSize,
                        BBType.GroundTruth,
                        format=bbFormat)
            else:
                if splitLine[0] != '':
                    idClass = (splitLine[0])
                    confidence = 1
                    # x = float(splitLine[1])
                    # y = float(splitLine[2])
                    # w = float(splitLine[3])
                    # h = float(splitLine[4])
                    x = float(splitLine[2])
                    y = float(splitLine[3])
                    w = float(splitLine[4])
                    h = float(splitLine[5])
                    bb = BoundingBox(
                        nameOfImage,
                        idClass,
                        x,
                        y,
                        w,
                        h,
                        coordType,
                        imgSize,
                        BBType.Detected,
                        confidence,
                        format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses


# Get current path to set default folders
currentPath = os.path.dirname(os.path.abspath(__file__))

VERSION = '0.1 (beta)'

parser = argparse.ArgumentParser(
    prog='Object Detection Metrics - Pascal VOC',
    description='This project applies the most popular metrics used to evaluate object detection '
                'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
                'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
    epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
# formatter_class=RawTextHelpFormatter)
parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
# Positional arguments
# Mandatory
parser.add_argument(
    '-gt',
    '--gtfolder',
    dest='gtFolder',
    default=GT_dir,
    metavar='',
    help='folder containing your ground truth bounding boxes')
parser.add_argument(
    '-det',
    '--detfolder',
    dest='detFolder',
    default=DT_dir,
    metavar='',
    help='folder containing your detected bounding boxes')
# Optional
parser.add_argument(
    '-t',
    '--threshold',
    dest='iouThreshold',
    type=float,
    default=0.5,
    metavar='',
    help='IOU threshold. Default 0.5')
parser.add_argument(
    '-gtformat',
    dest='gtFormat',
    metavar='',
    default='xywh',
    help='format of the coordinates of the ground truth bounding boxes: '
         '(\'xywh\': <left> <top> <width> <height>)'
         ' or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument(
    '-detformat',
    dest='detFormat',
    metavar='',
    default='xywh',
    help='format of the coordinates of the detected bounding boxes '
         '(\'xywh\': <left> <top> <width> <height>) '
         'or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument(
    '-gtcoords',
    dest='gtCoordinates',
    default='abs',
    metavar='',
    help='reference of the ground truth bounding box coordinates: absolute '
         'values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument(
    '-detcoords',
    default='abs',
    dest='detCoordinates',
    metavar='',
    help='reference of the ground truth bounding box coordinates: '
         'absolute values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument(
    '-imgsize',
    dest='imgSize',
    metavar='',
    help='image size. Required if -gtcoords or -detcoords are \'rel\'')
parser.add_argument(
    '-sp', '--savepath', dest='savePath', metavar='', help='folder where the plots are saved')
parser.add_argument(
    '-np',
    '--noplot',
    dest='showPlot',
    action='store_false',
    help='no plot is shown during execution')
args = parser.parse_args()

iouThreshold = args.iouThreshold

# Arguments validation
errors = []
# Validate formats
gtFormat = ValidateFormats(args.gtFormat, '-gtformat', errors)
detFormat = ValidateFormats(args.detFormat, '-detformat', errors)
# Groundtruth folder
if ValidateMandatoryArgs(args.gtFolder, '-gt/--gtfolder', errors):
    gtFolder = ValidatePaths(args.gtFolder, '-gt/--gtfolder', errors)
else:
    errors.pop()
    # gtFolder = os.path.join(currentPath, 'groundtruths')
    # if os.path.isdir(gtFolder) is False:
    #     errors.append('folder %s not found' % gtFolder)
# Coordinates types
gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
detCoordType = ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
imgSize = (0, 0)
if gtCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
if detCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)
# Detection folder
if ValidateMandatoryArgs(args.detFolder, '-det/--detfolder', errors):
    detFolder = ValidatePaths(args.detFolder, '-det/--detfolder', errors)
else:
    errors.pop()
    # detFolder = '/data/tuan-anh/aicr-TODAH/test_images/vn_test/Anno_txt'
    # if os.path.isdir(detFolder) is False:
    #     errors.append('folder %s not found' % detFolder)
if args.savePath is not None:
    savePath = ValidatePaths(args.savePath, '-sp/--savepath', errors)
else:
    savePath = os.path.join(currentPath, 'results_detector')
# Validate savePath
# If error, show error messages
if len(errors) is not 0:
    print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                [-detformat] [-save]""")
    print('Object Detection Metrics: error(s): ')
    [print(e) for e in errors]
    sys.exit()

# Create directory to save results
shutil.rmtree(savePath, ignore_errors=True)  # Clear folder
os.makedirs(savePath)

# Get groundtruth boxes
allBoundingBoxes, allClasses = getBoundingBoxes(
    gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize)
# Get detected boxes
allBoundingBoxes, allClasses = getBoundingBoxes(
    detFolder, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
# allClasses.sort()
evaluator = Evaluator()
# acc_AP = 0
# validClasses = 0

tf = evaluator.GetDetectorTruePositive(allBoundingBoxes)

TP_gt = tf['TP_gt']
FN_gt = tf['FN_gt']

print('class vn: ' + str(TP_gt / (TP_gt + FN_gt)) + ' TP: ' + str(TP_gt) + ' FP: ' + str(FN_gt))
