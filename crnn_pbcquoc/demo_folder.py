import torch, cv2
from torch.autograd import Variable
from torchvision import transforms, datasets
from loader import alignCollate
import models.crnn as crnn
from models.utils import strLabelConverter
import models.utils as utils

data_dir='/data/dataset/cinnamon_data'
imgW=512
imgH=32
batch_size = 1
alphabet_path='data/char_246'
alphabet = open(alphabet_path).read().rstrip()
nclass = len(alphabet) + 1
nc = 3
gpu=True
debug=True
workers=4
pretrained_model='outputs/train_2020-02-20_09-03/AICR_pretrained_60.pth'
image = torch.FloatTensor(batch_size, 3, imgH, imgH)
text = torch.IntTensor(batch_size * 5)
length = torch.IntTensor(batch_size)
model = crnn.CRNN(32, nc, nclass, 256)
if gpu:
    model = model.cuda()
    image = image.cuda()
print('loading pretrained model from %s' % pretrained_model)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

converter = strLabelConverter(alphabet, ignore_case=False)
data_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomErasing(),
    ])
data_transform = None
hymenoptera_dataset = datasets.ImageFolder(root=data_dir,
                                           transform=data_transform)
test_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=workers,
                                             collate_fn=alignCollate(imgW, imgH))

val_iter = iter(test_loader)
max_iter = len(test_loader)

with torch.no_grad():
    for i in range(max_iter):
        data = val_iter.next()
        cpu_images = data[0]
        utils.loadData(image, cpu_images)

        preds = model(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        print('\n    ', raw_pred, '\n =>', sim_pred)
        if debug:
            cv_img = cpu_images[0].permute(1, 2, 0).numpy()
            cv_img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            # cv_img_resize=cv2.resize(cv_img_bgr,cv_img_resize,)
            cv2.imshow('result', cv_img_bgr)
            cv2.waitKey(0)

