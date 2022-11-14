import argparse
import string

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms

from model import Model
from utils import TokenLabelConverter
from vit_grad_rollout import VITAttentionGradRollout
from vit_rollout import VITAttentionRollout

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')

    # vitstr argument
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='字符类别')
    choices = ['xvitstr_tiny_patch16_224',
               'xvitstr_small_patch16_224', 'xvitstr_base_patch16_224']
    parser.add_argument('--TransformerModel',
                        default=choices[0], help='哪一个vit模型', choices=choices)
    parser.add_argument('--saved_model', default='', help="继续训练的模型路径")
    parser.add_argument('--imgH', type=int, default=224, help='输入图片的高')
    parser.add_argument('--imgW', type=int, default=224, help='输入图片的宽')
    parser.add_argument('--batch_max_length', type=int,
                        default=25, help='标签长度的最大值')
    parser.add_argument('--img_path', default='', help='测试单张图像的路径')
    parser.add_argument('--isTrain', type=bool,
                        default=False, help='是否是训练模式。默认为训练模式')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap*0.7 + np.float32(img)*0.3
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == '__main__':
    args = get_args()
    opt = args
    opt.character = string.printable[:-6]
    cudnn.benchmark = True
    cudnn.deterministic = True
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # 打开图片
    img = Image.open(opt.img_path).convert('L')
    label = ''
    img_name = opt.img_path.replace(".jpg", "").split("/")[-1]
    # 调整大小
    image = img.resize((opt.imgW, opt.imgH), Image.BICUBIC)
    # 转为张量
    image_tensor = transforms.ToTensor()(image)
    # 调整维度
    image_tensor = image_tensor.unsqueeze(0)
    input_tensor = image_tensor

    if args.use_cuda:
        input_tensor = input_tensor.cuda()

    if args.category_index is None:
        print("Doing Attention Rollout")
        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)
        mask, mask_list = attention_rollout(input_tensor, converter)
        name = "{}_attention_rollout_{:.3f}_{}.jpg".format(img_name, args.discard_ratio, args.head_fusion)
    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        mask, mask_list = grad_rollout(input_tensor, args.category_index, converter)
        name = "{}_grad_rollout_{}_{:.3f}_{}.jpg".format(img_name, args.category_index, args.discard_ratio, args.head_fusion)
    # RGB
    # np_img = np.array(img)[:, :, ::-1]
    # 灰度图
    np_img = np.array(img)
    np_img = np.expand_dims(np_img, axis=2)
    
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    cv2.imwrite("./figures/" + name, mask)
    i = 0
    for mask1 in mask_list:
        mask2 = cv2.resize(mask1, (np_img.shape[1], np_img.shape[0]))
        mask2 = show_mask_on_image(np_img, mask2)
        cv2.imwrite("./figures/" + str(i) + name, mask2)
        i += 1
