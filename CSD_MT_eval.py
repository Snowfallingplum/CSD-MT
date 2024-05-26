import os
import torch
import cv2
import os.path as osp
import numpy as np
from PIL import Image
from CSD_MT.options import Options
from CSD_MT.model import CSD_MT
from faceutils.face_parsing.model import BiSeNet
import torchvision.transforms as transforms
import faceutils as futils


# load face_parsing model
n_classes = 19
face_paseing_model = BiSeNet(n_classes=n_classes)
save_pth = osp.join('faceutils/face_parsing/res/cp', '79999_iter.pth')
face_paseing_model.load_state_dict(torch.load(save_pth,map_location='cpu'))
face_paseing_model.eval()

# load makeup transfer model
parser = Options()
opts = parser.parse()
makeup_model = CSD_MT(opts)
ep0, total_it = makeup_model.resume('CSD_MT/weights/CSD_MT.pth')
makeup_model.eval()

def crop_image(image):
    up_ratio = 0.2 / 0.85  # delta_size / face_size
    down_ratio = 0.15 / 0.85  # delta_size / face_size
    width_ratio = 0.2 / 0.85  # delta_size / face_size

    image = Image.fromarray(image)
    face = futils.dlib.detect(image)

    if not face:
        raise ValueError("No face !")

    face_on_image = face[0]

    image, face, crop_face = futils.dlib.crop(image, face_on_image, up_ratio, down_ratio, width_ratio)
    np_image = np.array(image)
    return np_image

def get_face_parsing(x):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = Image.fromarray(x)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        out = face_paseing_model(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    return parsing


def split_parse(opts,parse):
    h, w = parse.shape
    result = np.zeros([h, w, opts.semantic_dim])
    result[:, :, 0][np.where(parse == 0)] = 1
    result[:, :, 0][np.where(parse == 16)] = 1
    result[:, :, 0][np.where(parse == 17)] = 1
    result[:, :, 0][np.where(parse == 18)] = 1
    result[:, :, 0][np.where(parse == 9)] = 1
    result[:, :, 1][np.where(parse == 1)] = 1
    result[:, :, 2][np.where(parse == 2)] = 1
    result[:, :, 2][np.where(parse == 3)] = 1
    result[:, :, 3][np.where(parse == 4)] = 1
    result[:, :, 3][np.where(parse == 5)] = 1
    result[:, :, 1][np.where(parse == 6)] = 1
    result[:, :, 4][np.where(parse == 7)] = 1
    result[:, :, 4][np.where(parse == 8)] = 1
    result[:, :, 5][np.where(parse == 10)] = 1
    result[:, :, 6][np.where(parse == 11)] = 1
    result[:, :, 7][np.where(parse == 12)] = 1
    result[:, :, 8][np.where(parse == 13)] = 1
    result[:, :, 9][np.where(parse == 14)] = 1
    result[:, :, 9][np.where(parse == 15)] = 1
    result = np.array(result)
    return result


def local_masks(opts,split_parse):
    h, w, c = split_parse.shape
    all_mask = np.zeros([h, w])
    all_mask[np.where(split_parse[:, :, 0] == 0)] = 1
    all_mask[np.where(split_parse[:, :, 3] == 1)] = 0
    all_mask[np.where(split_parse[:, :, 6] == 1)] = 0
    all_mask = np.expand_dims(all_mask, axis=2)  # Expansion of the dimension
    all_mask = np.concatenate((all_mask, all_mask, all_mask), axis=2)
    return all_mask



def load_data_from_image(non_makeup_img, makeup_img,opts):
    non_makeup_img=crop_image(non_makeup_img)
    makeup_img = crop_image(makeup_img)
    non_makeup_img=cv2.resize(non_makeup_img,(opts.resize_size,opts.resize_size))
    makeup_img = cv2.resize(makeup_img, (opts.resize_size, opts.resize_size))
    non_makeup_parse = get_face_parsing(non_makeup_img)
    non_makeup_parse = cv2.resize(non_makeup_parse, (opts.resize_size, opts.resize_size),interpolation=cv2.INTER_NEAREST)
    makeup_parse = get_face_parsing(makeup_img)
    makeup_parse = cv2.resize(makeup_parse, (opts.resize_size, opts.resize_size),interpolation=cv2.INTER_NEAREST)

    non_makeup_split_parse = split_parse(opts,non_makeup_parse)
    makeup_split_parse = split_parse(opts,makeup_parse)

    non_makeup_all_mask = local_masks(opts,
        non_makeup_split_parse)
    makeup_all_mask = local_masks(opts,
        makeup_split_parse)

    non_makeup_img = non_makeup_img / 127.5 - 1
    non_makeup_img = np.transpose(non_makeup_img, (2, 0, 1))
    non_makeup_split_parse = np.transpose(non_makeup_split_parse, (2, 0, 1))

    makeup_img = makeup_img / 127.5 - 1
    makeup_img = np.transpose(makeup_img, (2, 0, 1))
    makeup_split_parse = np.transpose(makeup_split_parse, (2, 0, 1))

    non_makeup_img=torch.from_numpy(non_makeup_img).type(torch.FloatTensor)
    non_makeup_img = torch.unsqueeze(non_makeup_img, 0)
    non_makeup_split_parse = torch.from_numpy(non_makeup_split_parse).type(torch.FloatTensor)
    non_makeup_split_parse = torch.unsqueeze(non_makeup_split_parse, 0)
    non_makeup_all_mask = np.transpose(non_makeup_all_mask, (2, 0, 1))

    makeup_img = torch.from_numpy(makeup_img).type(torch.FloatTensor)
    makeup_img = torch.unsqueeze(makeup_img, 0)
    makeup_split_parse = torch.from_numpy(makeup_split_parse).type(torch.FloatTensor)
    makeup_split_parse = torch.unsqueeze(makeup_split_parse, 0)
    makeup_all_mask = np.transpose(makeup_all_mask, (2, 0, 1))

    data = {'non_makeup_color_img': non_makeup_img,
            'non_makeup_split_parse':non_makeup_split_parse,
            'non_makeup_all_mask': torch.unsqueeze(torch.from_numpy(non_makeup_all_mask).type(torch.FloatTensor), 0),

            'makeup_color_img': makeup_img,
            'makeup_split_parse': makeup_split_parse,
            'makeup_all_mask': torch.unsqueeze(torch.from_numpy(makeup_all_mask).type(torch.FloatTensor), 0)
            }
    return data

def makeup_transfer256(non_makeup_image, makeup_image):
    data=load_data_from_image(non_makeup_image, makeup_image, opts=opts)
    with torch.no_grad():
        transfer_tensor=makeup_model.test_pair(data)
        transfer_img=transfer_tensor[0].cpu().float().numpy()
        transfer_img = np.transpose((transfer_img/ 2 + 0.5)*255., (1, 2, 0))
        transfer_img=np.clip(transfer_img+0.5,0,255).astype(np.uint8)
    return transfer_img